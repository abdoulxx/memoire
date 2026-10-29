# Image des conteneurs d'injection (livrable M4, phase C).
#
#   docker build -f infra/injector/Dockerfile -t memoire-chaos:0.18.07 .
#
# Construite par le depot et non tiree d'un registre tiers : un conteneur qui va
# consommer volontairement du CPU et de la memoire sur le serveur de Sidy ne doit
# contenir que ce qu'on y a mis. L'image ne contient qu'Alpine et stress-ng.
#
# La version du paquet est epinglee pour que la meme commande produise le meme
# outil de mesure : le cadrage demande que l'intensite d'une anomalie soit
# reproductible, et un stress-ng different ne charge pas de la meme facon.
FROM alpine:3.21

ARG STRESS_NG_VERSION=0.18.07-r0
RUN apk add --no-cache "stress-ng=${STRESS_NG_VERSION}"

# Version lisible depuis `docker inspect` : elle finit dans le memoire.
LABEL memoire.stress_ng_version="${STRESS_NG_VERSION}"

# Utilisateur non privilegie. stress-ng n'a besoin d'aucun privilege pour les
# deux seuls stresseurs utilises ici (--cpu et --vm) ; tout ce qui en demanderait
# serait hors du perimetre du §4.3.
RUN adduser -D -u 10002 chaos
USER chaos

# Repertoire de travail inscriptible, et ce n'est pas un detail de confort :
# stress-ng a besoin d'un `temp-path` accessible en ecriture et s'interrompt
# immediatement sans lui (`aborting: temp-path '.' must be readable and
# writeable`). Mesure a la verification de la phase C : les premieres injections
# s'arretaient a la premiere seconde, et l'injecteur ecrivait quand meme une
# etiquette — une anomalie annoncee qui n'avait pas eu lieu. D'ou cette ligne, et
# d'ou la verification de demarrage faite par l'injecteur.
WORKDIR /tmp

# Pas d'ENTRYPOINT : la commande complete est passee par l'injecteur, qui est le
# seul endroit ou les durees et les intensites sont decidees. Un ENTRYPOINT avec
# des valeurs par defaut permettrait de lancer une injection sans duree.
