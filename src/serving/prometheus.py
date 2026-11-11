"""Lecture de l'API Prometheus (livrable M4, phase D ; reutilise a M5).

Prometheus est la source du flux du mode en ligne (§2 du cadrage). Ce module est
le seul endroit du depot qui lui parle, et il fait une seule chose : rendre une
expression PromQL sous forme de serie reguliere, sur une grille d'instants
connue d'avance.

Trois proprietes, et chacune repond a un defaut qui fausserait les etiquettes :

1. **La grille est faite d'instants multiples du pas.** `query_range` evalue a
   `start`, `start + step`, ... : si `start` est arbitraire, la grille depend de
   l'heure a laquelle la commande a ete lancee, et deux enregistrements de la
   meme session ne tombent pas sur les memes instants. Le pas de collecte etant
   la minute, un decalage de quelques secondes suffit a deplacer une fenetre
   d'un cran par rapport a une anomalie etiquetee. Les instants sont donc
   arrondis au multiple du pas, et la grille est construite **avant** la
   requete, pas deduite de la reponse.

2. **Un trou reste un trou.** Quand une cible n'a pas repondu, Prometheus ne
   renvoie simplement pas de point : la reponse est plus courte, sans rien qui
   le signale. Les valeurs manquantes sont donc placees sur la grille comme des
   `NaN`, et c'est a l'appelant de decider ce qu'il en fait. Boucher un trou en
   silence par la valeur precedente inventerait une mesure, et une mesure
   inventee dans une fenetre etiquetee anormale est un faux resultat. Une valeur
   non finie (`NaN`, `+Inf`) compte aussi comme absente : elle est reelle du
   point de vue de Prometheus, mais un `inf` recopie dans une colonne traverse
   la normalisation et rend la session entiere inexploitable.

3. **Une expression qui rend plusieurs series est refusee.** Prendre la premiere
   serie serait un choix arbitraire fait par la bibliotheque a la place de
   l'experimentateur. L'erreur nomme les series rendues pour que l'expression
   soit corrigee (un `sum by (...)` manquant, le plus souvent).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np

# Prometheus refuse au-dela de 11 000 points par requete. Les plages plus
# longues sont decoupees ; le decoupage tombe sur des multiples du pas, donc les
# morceaux se recollent exactement.
POINTS_MAX_PAR_REQUETE = 10_000


class ErreurPrometheus(RuntimeError):
    """Prometheus a refuse la requete, ou la reponse est inutilisable."""


@dataclass(frozen=True)
class SerieTemporelle:
    """Une expression PromQL rendue sur une grille d'instants reguliere."""

    instants: np.ndarray
    """Secondes Unix, multiples du pas, strictement croissantes."""

    valeurs: np.ndarray
    """Valeurs, `NaN` la ou Prometheus n'avait pas de point."""

    def __len__(self) -> int:
        return len(self.instants)

    @property
    def part_manquante(self) -> float:
        """Part de points absents. Rapportee dans les metadonnees de la session."""
        if len(self.valeurs) == 0:
            return 1.0
        return float(np.isnan(self.valeurs).mean())


def grille(debut: float, fin: float, pas: int) -> np.ndarray:
    """Instants attendus entre `debut` et `fin`, tous multiples de `pas`.

    `debut` est arrondi vers le haut et `fin` vers le bas : la grille est
    toujours contenue dans la plage demandee, ce qui evite d'interroger
    Prometheus sur un instant futur au dernier point.
    """
    if pas < 1:
        raise ValueError("Le pas doit valoir au moins une seconde.")
    premier = math.ceil(debut / pas) * pas
    dernier = math.floor(fin / pas) * pas
    if dernier < premier:
        raise ValueError(
            f"Plage trop courte pour un pas de {pas} s : "
            f"aucun multiple du pas entre {debut:.0f} et {fin:.0f}."
        )
    return np.arange(premier, dernier + pas, pas, dtype=np.int64)


@dataclass
class ClientPrometheus:
    """Lecteur de l'API Prometheus. Lecture seule, aucune ecriture possible."""

    url: str
    delai_s: float = 60.0

    @property
    def base(self) -> str:
        return self.url.rstrip("/")

    def _appeler(self, chemin: str, parametres: dict[str, Any]) -> dict[str, Any]:
        try:
            reponse = httpx.get(f"{self.base}{chemin}", params=parametres, timeout=self.delai_s)
        except httpx.HTTPError as erreur:
            raise ErreurPrometheus(
                f"{self.base} n'a pas repondu : Prometheus est-il lance, et "
                f"PROMETHEUS_URL pointe-t-elle sur son port publie ? ({erreur})"
            ) from erreur
        # Une erreur PromQL arrive avec un code 400 et un corps JSON qui dit
        # pourquoi : le corps est bien plus utile que le code, donc il est lu
        # avant de lever.
        try:
            corps = reponse.json()
        except ValueError as erreur:
            raise ErreurPrometheus(
                f"{self.base}{chemin} a repondu {reponse.status_code} sans JSON exploitable."
            ) from erreur
        if corps.get("status") != "success":
            raise ErreurPrometheus(
                f"Prometheus a refuse la requete ({corps.get('errorType', 'erreur')}) : "
                f"{corps.get('error', 'sans message')}"
            )
        return dict(corps.get("data", {}))

    def interroger_plage(
        self, expression: str, debut: float, fin: float, pas: int
    ) -> SerieTemporelle:
        """Rend `expression` sur la grille des multiples de `pas` entre `debut` et `fin`."""
        instants = grille(debut, fin, pas)
        valeurs = np.full(len(instants), np.nan, dtype=np.float64)
        position = {int(t): i for i, t in enumerate(instants)}

        for depart in range(0, len(instants), POINTS_MAX_PAR_REQUETE):
            tranche = instants[depart : depart + POINTS_MAX_PAR_REQUETE]
            for instant, valeur in self._matrice(expression, tranche[0], tranche[-1], pas):
                indice = position.get(instant)
                if indice is not None:
                    valeurs[indice] = valeur

        return SerieTemporelle(instants=instants, valeurs=valeurs)

    def _matrice(self, expression: str, debut: int, fin: int, pas: int) -> list[tuple[int, float]]:
        donnees = self._appeler(
            "/api/v1/query_range",
            {"query": expression, "start": int(debut), "end": int(fin), "step": f"{pas}s"},
        )
        resultat = list(donnees.get("result", []))
        if len(resultat) > 1:
            etiquettes = [serie.get("metric", {}) for serie in resultat[:3]]
            raise ErreurPrometheus(
                f"L'expression rend {len(resultat)} series, une seule est attendue : "
                f"une session enregistree est une metrique par colonne. Agreger "
                f"l'expression (`sum by (...)`, `avg(...)`) ou la filtrer. "
                f"Exemples d'etiquettes rendues : {etiquettes}"
            )
        if not resultat:
            # Legitime : une metrique a zero serie (aucune 5xx, exportateur
            # absent) n'est pas une erreur ici. Le chargeur de session tranche,
            # parce que lui sait si la colonne est indispensable.
            return []
        points: list[tuple[int, float]] = []
        for instant, valeur in resultat[0].get("values", []):
            try:
                nombre = float(valeur)
            except (TypeError, ValueError):
                continue
            # Prometheus rend "NaN", "+Inf" et "-Inf" en texte, et `float` les
            # accepte sans broncher : un `histogram_quantile` sur un bucket vide
            # rend "NaN", une part calculee sur un denominateur nul rend "+Inf".
            # Ces valeurs doivent compter comme des points absents, pas etre
            # recopiees : un seul `inf` dans une colonne traverse la
            # normalisation et rend toute la session inexploitable, sans que rien
            # ne le signale.
            if not math.isfinite(nombre):
                continue
            points.append((int(instant), nombre))
        return points

    def cibles(self) -> dict[str, str]:
        """Etat des cibles de collecte : `{job: health}`.

        Sert de controle avant d'enregistrer une session : enregistrer pendant
        qu'une cible est tombee produirait des colonnes trouees, et le decouvrir
        apres coup coute une session entiere.
        """
        donnees = self._appeler("/api/v1/targets", {"state": "active"})
        etats: dict[str, str] = {}
        for cible in donnees.get("activeTargets", []):
            travail = str(cible.get("labels", {}).get("job", "inconnu"))
            sante = str(cible.get("health", "unknown"))
            # Si plusieurs instances portent le meme job, la pire l'emporte :
            # c'est l'etat qui compte pour la validite de la session.
            if etats.get(travail) != "down":
                etats[travail] = sante
        return etats
