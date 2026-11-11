"""Signature du contrat de features d'un modele.

Un modele entraine a un instant donne attend **exactement** les features sur
lesquelles il a ete ajuste : les memes metriques, dans le meme ordre, avec les
memes agregats de fenetre. Rien dans un `IsolationForest` ne verifie cela.
Scorer une matrice dont les colonnes ont change de sens ne declenche aucune
erreur tant que leur nombre est le meme : le modele rend des scores plausibles,
calcules sur autre chose que ce qu'il a appris.

En rejeu, le risque n'existe pas : les features sont construites dans le meme
processus que l'entrainement. En mode en ligne, le champion est charge depuis le
registre, parfois des jours apres avoir ete entraine, et la configuration de la
source a pu changer entre-temps — une metrique ajoutee a
`configs/datasets/serveur.yaml`, un agregat retire de `configs/features.yaml`,
une taille de fenetre modifiee. C'est precisement le genre de changement qu'on
fait sans y penser, et il rendrait toutes les alertes du service fausses sans
aucun signe visible.

La signature est donc enregistree avec chaque version de modele et **verifiee au
chargement**. Un ecart fait echouer le demarrage du service : mieux vaut un
service qui refuse de partir qu'un service qui alerte au hasard.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SignatureFeatures:
    """Ce qu'un modele attend en entree, sous une forme comparable et journalisable.

    `empreinte` resume les noms de features **et** les parametres de fenetrage.
    Deux signatures egales garantissent que la colonne *j* a le meme sens dans
    les deux cas ; une difference, meme sur un seul agregat, donne une empreinte
    differente.
    """

    feature_names: tuple[str, ...]
    window_size: int
    stride: int
    sampling_interval_seconds: int

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def empreinte(self) -> str:
        """Condense la signature en 16 caracteres hexadecimaux, stables entre executions.

        `sha256` et non `hash()` : le hachage de Python est randomise a chaque
        processus, donc inutilisable pour comparer a une valeur enregistree hier.
        """
        charge = json.dumps(
            {
                "feature_names": list(self.feature_names),
                "window_size": self.window_size,
                "stride": self.stride,
                "sampling_interval_seconds": self.sampling_interval_seconds,
            },
            ensure_ascii=True,
            sort_keys=False,
        )
        return hashlib.sha256(charge.encode("utf-8")).hexdigest()[:16]

    def as_tags(self) -> dict[str, str]:
        """Forme enregistree a cote du modele. Les valeurs sont des chaines (contrainte MLflow)."""
        return {
            "features_empreinte": self.empreinte,
            "features_n": str(self.n_features),
            "features_window_size": str(self.window_size),
            "features_stride": str(self.stride),
            "features_pas_s": str(self.sampling_interval_seconds),
        }

    def ecart(self, autre: SignatureFeatures) -> str:
        """Decrit en une phrase ce qui differe, ou une chaine vide si les deux concordent.

        Le message nomme la cause probable : un service qui refuse de demarrer
        doit dire quoi corriger, sinon on le relance en boucle.
        """
        if self.empreinte == autre.empreinte:
            return ""
        if self.n_features != autre.n_features:
            return (
                f"{autre.n_features} features attendues par le modele, "
                f"{self.n_features} produites par la configuration actuelle. "
                "Une metrique a ete ajoutee ou retiree de la source, ou un agregat "
                "de configs/features.yaml a change."
            )
        if (self.window_size, self.stride) != (autre.window_size, autre.stride):
            return (
                f"Fenetrage different : le modele a ete entraine avec "
                f"window_size={autre.window_size}, stride={autre.stride} ; la "
                f"configuration actuelle donne window_size={self.window_size}, "
                f"stride={self.stride}."
            )
        if self.sampling_interval_seconds != autre.sampling_interval_seconds:
            return (
                f"Pas d'echantillonnage different : {autre.sampling_interval_seconds}s "
                f"a l'entrainement, {self.sampling_interval_seconds}s maintenant. "
                "Une fenetre ne couvre plus la meme duree."
            )
        differents = [
            f"colonne {i} : « {b} » attendue, « {a} » produite"
            for i, (a, b) in enumerate(zip(self.feature_names, autre.feature_names, strict=True))
            if a != b
        ]
        return (
            f"{len(differents)} colonne(s) ont change de sens, a nombre egal. "
            + " ; ".join(differents[:3])
            + ("..." if len(differents) > 3 else "")
        )


def signature_depuis_config(
    metric_names: list[str],
    features_config: dict[str, Any],
    sampling_interval_seconds: int,
) -> SignatureFeatures:
    """Reconstruit la signature que `make_windows` produira pour ces metriques.

    L'ordre est celui de `make_windows` — agregat par agregat, puis metrique par
    metrique a l'interieur de chaque bloc. Les deux doivent rester d'accord ;
    un test le verifie en comparant a `WindowedData.feature_names`.
    """
    aggregates = list(features_config["aggregates"])
    noms = tuple(f"{metrique}_{agregat}" for agregat in aggregates for metrique in metric_names)
    return SignatureFeatures(
        feature_names=noms,
        window_size=int(features_config["window_size"]),
        stride=int(features_config["stride"]),
        sampling_interval_seconds=int(sampling_interval_seconds),
    )
