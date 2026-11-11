"""Registre de modeles : traçabilite des versions et du modele en service.

Role dans le memoire (§3) : « traçabilite des versions et des reentrainements ».
Le registre repond a une question que le journal d'evenements ne suffit pas a
trancher : **quel modele, exactement, a produit cette alerte ?** Une ligne
`promoted` dit qu'une promotion a eu lieu ; elle ne permet pas de recharger le
modele promu trois semaines plus tard pour l'expliquer a un jury.

Quatre decisions portent ce module.

**1. Le modele enregistre est le detecteur entier, pas la foret.**
`mlflow.sklearn` sait enregistrer un `IsolationForest`, mais le modele en
service n'est pas la foret : c'est le triplet standardisation + foret + seuil
(`src/models/isolation_forest.py`), et le §4.4 impose que la normalisation soit
sauvegardee avec le modele. Charger la seule foret depuis un flavour sklearn
donnerait un objet sans seuil ni standardisation — chaque score serait compare a
un seuil venu d'ailleurs, et rien ne le signalerait. L'artefact enregistre est
donc le fichier joblib que `IsolationForestDetector.save` produit deja, celui-la
meme que le harnais de rejeu relit : le chapitre 5 et le chapitre 6 chargent le
modele par le meme chemin de code.

**2. La signature des features est enregistree et verifiee au chargement.**
Voir `src/features/signature.py` : un champion charge dont le contrat de
features ne correspond plus a la configuration courante fait **echouer** le
chargement.

**3. Un registre injoignable ne doit pas suspendre la detection, mais ne doit
pas non plus etre contourne.** Au demarrage, un registre muet fait echouer le
service : servir un modele dont personne ne connait la version rendrait le
tableau de bord et les captures du chapitre 5 mensongers. En cours de route, une
promotion qui ne peut pas etre enregistree est **refusee** : le champion reste en
service — ce qui degrade la strategie vers B0 pour un temps, pas plus — et la
cause est journalisee. L'inverse (promouvoir sans enregistrer) ferait servir un
modele non identifiable, donc non explicable.

**4. Les delais du client sont bornes.** Par defaut, mlflow attend 120 s par
requete et retente plusieurs fois : un registre tombe bloquerait la boucle
pendant des minutes et lui ferait manquer des fenetres, alors que l'intervalle
de collecte est d'une minute. Les bornes viennent de la configuration.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.features.signature import SignatureFeatures
from src.models.isolation_forest import IsolationForestDetector

if TYPE_CHECKING:  # pragma: no cover - uniquement pour les annotations
    from mlflow.entities.model_registry import ModelVersion

ALIAS_CHAMPION = "champion"
NOM_ARTEFACT = "modele"
FICHIER_MODELE = "detecteur.joblib"

# Les features d'entrainement du modele, enregistrees a cote de lui. Elles ne
# servent pas a predire : elles sont la **reference** du moniteur de derive
# (`DriftMonitor.fit_reference`), qui compare la distribution recente a celle
# sur laquelle le modele en service a appris. Sans elles, un service qui
# redemarre ne peut que recaler son moniteur sur les donnees recentes, c'est-a-dire
# comparer le present a lui-meme : une derive survenue pendant l'arret devient
# invisible, definitivement, et rien ne le signale. C'est le genre de defaut
# qui ne se voit sur aucun tableau de bord.
NOM_ARTEFACT_REFERENCE = "reference"
FICHIER_REFERENCE = "features_entrainement.npz"


class ErreurRegistre(RuntimeError):
    """Le registre ne peut pas repondre, ou ce qu'il contient n'est pas utilisable."""


@dataclass(frozen=True)
class ReglagesRegistre:
    """Configuration du registre, lue dans `configs/serving.yaml`."""

    tracking_uri: str
    experiment: str
    model_name: str
    alias: str = ALIAS_CHAMPION
    timeout_s: float = 10.0
    max_retries: int = 1
    # Ou le registre depose les artefacts, utilise **uniquement** a la creation
    # de l'experience. Avec un serveur mlflow, c'est le serveur qui decide et ce
    # champ reste vide ; avec un backend sqlite local, mlflow ecrirait sinon dans
    # un `mlruns/` cree a l'endroit d'ou la commande a ete lancee.
    artifact_location: str | None = None

    @classmethod
    def depuis_config(
        cls, config: dict[str, Any], env: dict[str, str] | None = None
    ) -> ReglagesRegistre:
        bloc = dict(config.get("registry", {}) or {})
        if not bloc:
            raise ErreurRegistre(
                "Section `registry:` absente de la configuration du mode en ligne. "
                "Voir configs/serving.yaml."
            )
        environnement = os.environ if env is None else env
        # Meme regle que pour PROMETHEUS_URL (phase D) : la variable
        # d'environnement a la priorite, parce que l'URL depend du deploiement et
        # n'a rien a faire dans un fichier suivi par git.
        uri = environnement.get("MLFLOW_TRACKING_URI") or str(bloc.get("tracking_uri", ""))
        if not uri:
            raise ErreurRegistre(
                "Aucune URI de suivi : renseigner MLFLOW_TRACKING_URI dans .env ou "
                "`registry.tracking_uri` dans configs/serving.yaml."
            )
        timeout = float(bloc.get("timeout_s", 10.0))
        if timeout <= 0:
            raise ErreurRegistre(f"`registry.timeout_s` doit etre > 0, recu {timeout}.")
        retries = int(bloc.get("max_retries", 1))
        if retries < 0:
            raise ErreurRegistre(f"`registry.max_retries` doit etre >= 0, recu {retries}.")
        return cls(
            tracking_uri=uri,
            experiment=str(bloc.get("experiment", "detection-anomalies")),
            model_name=str(bloc.get("model_name", "detecteur-anomalies")),
            alias=str(bloc.get("alias", ALIAS_CHAMPION)),
            timeout_s=timeout,
            max_retries=retries,
            artifact_location=(
                str(bloc["artifact_location"]) if bloc.get("artifact_location") else None
            ),
        )


@dataclass(frozen=True)
class VersionModele:
    """Une version du modele telle que le registre la decrit.

    `signature` est celle qui a ete enregistree a l'entrainement : c'est elle que
    le chargement compare a la configuration courante.
    """

    name: str
    version: int
    run_id: str
    threshold: float
    n_train_windows: int
    raison: str
    cree_le: str
    alias: str | None = None
    signature: SignatureFeatures | None = None
    tags: dict[str, str] | None = None
    # Nombre de fenetres de reference enregistrees avec le modele (0 = aucune).
    reference_n_windows: int = 0

    @property
    def a_une_reference(self) -> bool:
        return self.reference_n_windows > 0

    @property
    def est_champion(self) -> bool:
        return self.alias == ALIAS_CHAMPION

    def resume(self) -> str:
        """Une ligne lisible, pour le journal d'evenements et les messages Telegram."""
        marque = f" [{self.alias}]" if self.alias else ""
        return (
            f"{self.name} v{self.version}{marque} seuil={self.threshold:.4f} "
            f"entraine sur {self.n_train_windows} fenetres ({self.raison})"
        )


def _maintenant() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class RegistreModeles:
    """Acces au suivi et au registre MLflow, reduit a ce dont la boucle a besoin.

    L'import de mlflow est differe : le socle ML du memoire se reproduit sans le
    groupe `mlops`, et importer un serveur web pour rejouer le SMD n'aurait pas
    de sens. Une installation sans mlflow echoue ici, avec un message qui dit
    quoi installer, et nulle part ailleurs.
    """

    def __init__(self, reglages: ReglagesRegistre) -> None:
        self.reglages = reglages
        self._client: Any = None

    # ------------------------------------------------------------------ client

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._construire_client()
        return self._client

    def _construire_client(self) -> Any:
        try:
            from mlflow.tracking import MlflowClient
        except ImportError as erreur:  # pragma: no cover - depend de l'installation
            raise ErreurRegistre(
                "mlflow n'est pas installe. Le mode en ligne demande le groupe "
                "`mlops` : uv sync --group mlops."
            ) from erreur

        # Bornes des requetes HTTP. mlflow ne les lit que dans l'environnement ;
        # elles sont posees avant la construction du client, et seulement si
        # l'exploitant ne les a pas deja fixees lui-meme.
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", str(int(self.reglages.timeout_s)))
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", str(self.reglages.max_retries))
        return MlflowClient(tracking_uri=self.reglages.tracking_uri)

    def _erreur(self, action: str, erreur: Exception) -> ErreurRegistre:
        return ErreurRegistre(
            f"{action} a echoue sur le registre {self.reglages.tracking_uri} : {erreur}. "
            "Verifier que le service mlflow tourne (docker compose ps) et que "
            "MLFLOW_TRACKING_URI pointe dessus."
        )

    def verifier(self) -> None:
        """Verifie que le registre repond. A appeler avant de demarrer le service."""
        try:
            self.client.search_experiments(max_results=1)
        except Exception as erreur:  # noqa: BLE001 - toute panne de transport compte
            raise self._erreur("La verification du registre", erreur) from erreur

    # ------------------------------------------------------------- ecriture

    def _experiment_id(self) -> str:
        existant = self.client.get_experiment_by_name(self.reglages.experiment)
        if existant is not None:
            return str(existant.experiment_id)
        return str(
            self.client.create_experiment(
                self.reglages.experiment, artifact_location=self.reglages.artifact_location
            )
        )

    def _assurer_modele(self) -> None:
        # `get_registered_model` leve quand le modele n'existe pas encore : c'est
        # l'etat d'un registre neuf, pas une panne. On ne cree qu'en reponse a
        # cette absence precise, et toute autre erreur remonte.
        try:
            self.client.get_registered_model(self.reglages.model_name)
        except Exception as erreur:  # noqa: BLE001
            if not self._absence(erreur):
                raise
            self.client.create_registered_model(self.reglages.model_name)

    def enregistrer(
        self,
        detecteur: IsolationForestDetector,
        *,
        signature: SignatureFeatures,
        raison: str,
        parametres: dict[str, Any] | None = None,
        metriques: dict[str, float] | None = None,
        tags: dict[str, str] | None = None,
        reference_X: np.ndarray | None = None,
    ) -> VersionModele:
        """Enregistre un modele entraine et rend la version creee.

        La version n'est **pas** promue : `promouvoir` est un appel separe, parce
        que l'enregistrement a lieu avant la validation et que la plupart des
        challengers d'une campagne sont rejetes. Un registre qui ne contiendrait
        que les modeles promus ne permettrait pas d'expliquer un rejet.
        """
        if not detecteur.is_fitted:
            raise ErreurRegistre("Modele non entraine : rien a enregistrer.")

        try:
            experiment_id = self._experiment_id()
            run = self.client.create_run(
                experiment_id=experiment_id,
                run_name=f"{raison}-{_maintenant()}",
                tags={"raison": raison, **signature.as_tags(), **(tags or {})},
            )
            run_id = str(run.info.run_id)

            for cle, valeur in (parametres or {}).items():
                self.client.log_param(run_id, cle, valeur)
            self.client.log_param(run_id, "threshold_percentile", detecteur.threshold_percentile)
            self.client.log_param(run_id, "scaling", detecteur.scaling)
            self.client.log_metric(run_id, "threshold", float(detecteur.threshold))
            self.client.log_metric(run_id, "n_train_windows", float(detecteur.n_train_windows))
            self.client.log_metric(run_id, "fit_seconds", float(detecteur.fit_seconds))
            for cle, valeur in (metriques or {}).items():
                self.client.log_metric(run_id, cle, float(valeur))

            with tempfile.TemporaryDirectory() as dossier:
                chemin = detecteur.save(Path(dossier) / FICHIER_MODELE)
                self.client.log_artifact(run_id, str(chemin), artifact_path=NOM_ARTEFACT)
                if reference_X is not None:
                    # Compresse : W fenetres x 152 features en float64 font
                    # quelques mega-octets par version, et une campagne en
                    # produit plusieurs dizaines.
                    reference = Path(dossier) / FICHIER_REFERENCE
                    np.savez_compressed(reference, features=np.asarray(reference_X))
                    self.client.log_artifact(
                        run_id, str(reference), artifact_path=NOM_ARTEFACT_REFERENCE
                    )
            self.client.set_terminated(run_id, "FINISHED")

            self._assurer_modele()
            source = f"{run.info.artifact_uri}/{NOM_ARTEFACT}"
            version = self.client.create_model_version(
                name=self.reglages.model_name,
                source=source,
                run_id=run_id,
                tags={
                    "raison": raison,
                    "seuil": f"{float(detecteur.threshold):.6f}",
                    "n_train_windows": str(detecteur.n_train_windows),
                    # 0 dit « cette version n'a pas de reference » : le service
                    # le lit avant de telecharger, et refuse de demarrer sur un
                    # champion qu'il ne saurait pas surveiller.
                    "reference_n_windows": str(
                        0 if reference_X is None else len(np.asarray(reference_X))
                    ),
                    **signature.as_tags(),
                    **(tags or {}),
                },
            )
        except ErreurRegistre:
            raise
        except Exception as erreur:  # noqa: BLE001
            raise self._erreur("L'enregistrement du modele", erreur) from erreur

        return self._decrire(version)

    def promouvoir(self, version: VersionModele | int) -> VersionModele:
        """Place l'alias `champion` sur cette version. C'est la mise en service."""
        numero = version.version if isinstance(version, VersionModele) else int(version)
        try:
            self.client.set_registered_model_alias(
                self.reglages.model_name, self.reglages.alias, str(numero)
            )
            decrite = self.client.get_model_version(self.reglages.model_name, str(numero))
        except Exception as erreur:  # noqa: BLE001
            raise self._erreur(f"La promotion de la version {numero}", erreur) from erreur
        return self._decrire(decrite, alias=self.reglages.alias)

    # ------------------------------------------------------------- lecture

    def champion(self) -> VersionModele | None:
        """La version portant l'alias, ou None si aucune n'a encore ete promue."""
        try:
            version = self.client.get_model_version_by_alias(
                self.reglages.model_name, self.reglages.alias
            )
        except Exception as erreur:  # noqa: BLE001
            # Un modele jamais cree ou un alias jamais pose ne sont pas des
            # pannes : c'est l'etat d'un registre neuf, et l'appelant doit
            # pouvoir le distinguer d'un registre injoignable. On verifie donc
            # que le transport repond avant de conclure a l'absence.
            self.verifier()
            if self._absence(erreur):
                return None
            raise self._erreur("La lecture du champion", erreur) from erreur
        return self._decrire(version, alias=self.reglages.alias)

    @staticmethod
    def _absence(erreur: Exception) -> bool:
        texte = str(erreur).lower()
        return any(
            marque in texte
            for marque in ("not found", "does not exist", "resource_does_not_exist", "no alias")
        )

    def historique(self, limite: int = 20) -> list[VersionModele]:
        """Les versions, de la plus recente a la plus ancienne."""
        try:
            versions = self.client.search_model_versions(
                f"name = '{self.reglages.model_name}'",
                max_results=limite,
                order_by=["version_number DESC"],
            )
        except Exception as erreur:  # noqa: BLE001
            self.verifier()
            if self._absence(erreur):
                return []
            raise self._erreur("La lecture de l'historique", erreur) from erreur
        courant = self.champion()
        numero_champion = courant.version if courant else None
        return [
            self._decrire(
                version,
                alias=self.reglages.alias if int(version.version) == numero_champion else None,
            )
            for version in versions
        ]

    def charger(
        self, version: VersionModele | int, signature_attendue: SignatureFeatures | None = None
    ) -> tuple[IsolationForestDetector, VersionModele]:
        """Telecharge et deserialise une version. Refuse un contrat de features different."""
        numero = version.version if isinstance(version, VersionModele) else int(version)
        try:
            decrite = self._decrire(
                self.client.get_model_version(self.reglages.model_name, str(numero))
            )
        except Exception as erreur:  # noqa: BLE001
            raise self._erreur(f"La lecture de la version {numero}", erreur) from erreur

        if signature_attendue is not None:
            if decrite.signature is None:
                raise ErreurRegistre(
                    f"La version {numero} ne porte aucune signature de features : elle a ete "
                    "enregistree hors de ce code. Impossible de verifier qu'elle attend les "
                    "memes colonnes, donc impossible de la mettre en service."
                )
            ecart = signature_attendue.ecart(decrite.signature)
            if ecart:
                raise ErreurRegistre(
                    f"La version {numero} n'attend pas les features produites par la "
                    f"configuration actuelle. {ecart} Reentrainer un modele initial "
                    "(mdrift registry-init) ou revenir a la configuration d'origine."
                )

        with tempfile.TemporaryDirectory() as dossier:
            try:
                chemin = self.client.download_artifacts(
                    decrite.run_id, f"{NOM_ARTEFACT}/{FICHIER_MODELE}", dossier
                )
            except Exception as erreur:  # noqa: BLE001
                raise self._erreur(f"Le telechargement de la version {numero}", erreur) from erreur
            detecteur = IsolationForestDetector.load(chemin)

        if not detecteur.is_fitted:
            raise ErreurRegistre(
                f"La version {numero} contient un detecteur non entraine : inutilisable."
            )
        return detecteur, decrite

    def charger_champion(
        self, signature_attendue: SignatureFeatures | None = None
    ) -> tuple[IsolationForestDetector, VersionModele]:
        """Charge le modele en service. Echoue s'il n'y en a pas : c'est voulu."""
        courant = self.champion()
        if courant is None:
            raise ErreurRegistre(
                f"Aucun modele ne porte l'alias `{self.reglages.alias}` dans "
                f"{self.reglages.model_name}. Entrainer et promouvoir un modele initial "
                "avec `mdrift registry-init` avant de demarrer le service."
            )
        detecteur, decrite = self.charger(courant, signature_attendue)
        return detecteur, replace(decrite, alias=self.reglages.alias)

    def charger_reference(self, version: VersionModele | int) -> np.ndarray:
        """Telecharge les features d'entrainement d'une version.

        Echoue explicitement quand la version n'en porte pas : demarrer le
        service sans reference reviendrait a recaler le moniteur de derive sur
        les donnees du moment, donc a comparer le present a lui-meme. Mieux vaut
        refuser de demarrer et le dire.
        """
        if isinstance(version, VersionModele):
            decrite = version
        else:
            try:
                decrite = self._decrire(
                    self.client.get_model_version(self.reglages.model_name, str(int(version)))
                )
            except Exception as erreur:  # noqa: BLE001
                raise self._erreur(f"La lecture de la version {version}", erreur) from erreur
        if not decrite.a_une_reference:
            raise ErreurRegistre(
                f"La version {decrite.version} n'a pas de features d'entrainement "
                "enregistrees : le moniteur de derive n'aurait aucune reference a "
                "laquelle comparer les fenetres recentes. Reentrainer un modele initial "
                "avec `mdrift registry-init`."
            )
        with tempfile.TemporaryDirectory() as dossier:
            try:
                chemin = self.client.download_artifacts(
                    decrite.run_id, f"{NOM_ARTEFACT_REFERENCE}/{FICHIER_REFERENCE}", dossier
                )
            except Exception as erreur:  # noqa: BLE001
                raise self._erreur(
                    f"Le telechargement de la reference de la version {decrite.version}", erreur
                ) from erreur
            with np.load(chemin) as archive:
                features = np.asarray(archive["features"], dtype=np.float64)
        if features.ndim != 2 or len(features) == 0:
            raise ErreurRegistre(
                f"La reference de la version {decrite.version} a la forme {features.shape} : "
                "une matrice (fenetres, features) est attendue."
            )
        return features

    # ------------------------------------------------------------- conversion

    def _decrire(self, version: ModelVersion, alias: str | None = None) -> VersionModele:
        tags = dict(getattr(version, "tags", {}) or {})
        signature = _signature_depuis_tags(tags)
        cree = getattr(version, "creation_timestamp", None)
        return VersionModele(
            name=str(version.name),
            version=int(version.version),
            run_id=str(version.run_id),
            threshold=float(tags.get("seuil", "nan")),
            n_train_windows=int(tags.get("n_train_windows", 0)),
            raison=str(tags.get("raison", "")),
            reference_n_windows=int(tags.get("reference_n_windows", 0)),
            cree_le=(
                datetime.fromtimestamp(cree / 1000, UTC).isoformat(timespec="seconds")
                if cree
                else ""
            ),
            alias=alias
            or (self.reglages.alias if _porte_alias(version, self.reglages.alias) else None),
            signature=signature,
            tags=tags,
        )


def _porte_alias(version: ModelVersion, alias: str) -> bool:
    return alias in tuple(getattr(version, "aliases", ()) or ())


def _signature_depuis_tags(tags: dict[str, str]) -> SignatureFeatures | None:
    """Reconstruit la signature enregistree, ou None si la version n'en porte pas.

    Les noms de features ne sont pas stockes (ils feraient une etiquette de
    plusieurs kilo-octets) : seule l'empreinte l'est. La signature reconstruite
    porte donc un unique nom factice qui **contient** l'empreinte, de sorte que
    la comparaison par empreinte reste exacte et que le message d'ecart reste
    lisible.
    """
    empreinte = tags.get("features_empreinte")
    if not empreinte:
        return None
    return _SignatureEnregistree(
        feature_names=tuple(f"feature_{i:03d}" for i in range(int(tags.get("features_n", 0)))),
        window_size=int(tags.get("features_window_size", 0)),
        stride=int(tags.get("features_stride", 0)),
        sampling_interval_seconds=int(tags.get("features_pas_s", 0)),
        empreinte_enregistree=empreinte,
    )


@dataclass(frozen=True)
class _SignatureEnregistree(SignatureFeatures):
    """Signature relue depuis les etiquettes : son empreinte est celle qui a ete enregistree.

    Recalculer l'empreinte a partir de noms factices donnerait une valeur qui ne
    correspondrait a rien. La sous-classe conserve donc l'empreinte telle qu'elle
    a ete ecrite a l'entrainement, et le reste de la comparaison (nombre de
    features, fenetrage) fonctionne a l'identique.
    """

    empreinte_enregistree: str = ""

    @property
    def empreinte(self) -> str:
        return self.empreinte_enregistree
