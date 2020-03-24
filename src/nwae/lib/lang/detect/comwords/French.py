# --*-- coding: utf-8 --*--

from nwae.utils.Log import Log
from inspect import getframeinfo, currentframe
from nwae.lib.lang.LangFeatures import LangFeatures
from nwae.lib.lang.detect.CommonWords import CommonWords


class French(CommonWords):

    # We assume 20% as minimum
    MIN_FR_SENT_INTERSECTION_PCT = 0.2

    def __init__(
            self
    ):
        super().__init__(lang=LangFeatures.LANG_FR)

        self.raw_words = """
comme
je
son
que
il
était
pour
sur
sont
avec
ils
être
à
un
avoir
ce
à partir de
par
chaud
mot
mais
que
certains
est
il
vous
ou
eu
la
de
à
et
un
dans
nous
boîte
dehors
autre
étaient
qui
faire
leur
temps
si
volonté
comment
dit
un
chaque
dire
ne
ensemble
trois
vouloir
air
bien
aussi
jouer
petit
fin
mettre
maison
lire
main
port
grand
épeler
ajouter
même
terre
ici
il faut
grand
haut
tel
suivre
acte
pourquoi
interroger
hommes
changement
est allé
lumière
genre
de
besoin
maison
image
essayer
nous
encore
animal
point
mère
monde
près de
construire
soi
terre
père
tout
nouveau
travail
partie
prendre
obtenir
lieu
fabriqué
vivre
où
après
arrière
peu
seulement
tour
homme
année
est venu
montrer
tous
bon
moi
donner
notre
sous
nom
très
par
juste
forme
phrase
grand
penser
dire
aider
faible
ligne
différer
tour
la cause
beaucoup
signifier
avant
déménagement
droit
garçon
vieux
trop
même
elle
tous
là
quand
jusqu’à
utiliser
votre
manière
sur
beaucoup
puis
les
écrire
voudrais
comme
si
ces
son
long
faire
chose
voir
lui
deux
a
regarder
plus
jour
pourrait
aller
venir
fait
nombre
son
aucun
plus
personnes
ma
sur
savoir
eau
que
appel
première
qui
peut
vers le bas
côté
été
maintenant
trouver
tête
supporter
propre
page
devrait
pays
trouvé
réponse
école
croître
étude
encore
apprendre
usine
couvercle
nourriture
soleil
quatre
entre
état
garder
œil
jamais
dernier
laisser
pensée
ville
arbre
traverser
ferme
dur
début
puissance
histoire
scie
loin
mer
tirer
gauche
tard
courir
needs a context
tandis que
presse
proche
nuit
réel
vie
peu
nord
livre
porter
a pris
science
manger
chambre
ami
a commencé
idée
poisson
montagne
Arrêtez
une fois
base
entendre
cheval
coupe
sûr
regarder
couleur
face
bois
principal
ouvert
paraître
ensemble
suivant
blanc
enfants
commencer
eu
marcher
exemple
facilité
papier
groupe
toujours
musique
ceux
tous les deux
marque
souvent
lettre
jusqu’à ce que
mile
rivière
voiture
pieds
soins
deuxième
assez
plaine
fille
habituel
jeune
prêt
au-dessus
jamais
rouge
liste
bien que
sentir
parler
oiseau
bientôt
corps
chien
famille
direct
pose
laisser
chanson
mesurer
porte
produit
noir
court
chiffre
classe
vent
question
arriver
complète
navire
zone
moitié
rock
ordre
feu
sud
problème
pièce
dit
savait
passer
depuis
haut
ensemble
roi
rue
pouce
multiplier
rien
cours
rester
roue
plein
force
bleu
objet
décider
surface
profond
lune
île
pied
système
occupé
test
record
bateau
commun
or
possible
plan
place
sec
se demander
rire
mille
il ya
ran
vérifier
jeu
forme
assimiler
chaud
manquer
apporté
chaleur
neige
pneu
apporter
oui
lointain
remplir
est
peindre
langue
entre
unité
puissance
ville
fin
certain
voler
tomber
conduire
cri
sombre
machine
Note
patienter
plan
figure
étoile
boîte
nom
domaine
reste
correct
capable
livre
Terminé
beauté
entraînement
résisté
contenir
avant
enseigner
semaine
finale
donné
vert
oh
rapide
développer
océan
chaud
gratuit
minute
fort
spécial
esprit
derrière
clair
queue
produire
fait
espace
entendu
meilleur
heure
mieux
vrai
pendant
cent
cinq
rappeler
étape
tôt
tenir
ouest
sol
intérêt
atteindre
rapide
verbe
chanter
écouter
six
table
Voyage
moins
matin
dix
simple
plusieurs
voyelle
vers
guerre
poser
contre
modèle
lent
centre
amour
personne
argent
servir
apparaître
route
carte
pluie
règle
gouverner
tirer
froid
avis
voix
énergie
chasse
probable
lit
frère
œuf
tour
cellule
croire
peut-être
choisir
soudain
compter
carré
raison
longueur
représenter
art
sujet
région
taille
varier
régler
parler
poids
général
glace
question
cercle
paire
inclure
fracture
syllabe
feutre
grandiose
balle
encore
vague
tomber
cœur
h
présent
lourd
danse
moteur
position
bras
large
voile
matériel
fraction
forêt
s’asseoir
course
fenêtre
magasin
été
train
sommeil
prouver
seul
jambe
exercice
mur
capture
monture
souhaiter
ciel
conseil
joie
hiver
sat
écrit
sauvage
instrument
conservé
verre
herbe
vache
emploi
bord
signe
visite
passé
doux
amusement
clair
gaz
temps
mois
million
porter
finition
heureux
espoir
fleur
vêtir
étrange
disparu
commerce
mélodie
voyage
bureau
recevoir
rangée
bouche
exact
symbole
mourir
moins
difficulté
cri
sauf
écrit
semence
ton
joindre
suggérer
propre
pause
dame
cour
augmenter
mauvais
coup
huile
sang
toucher
a augmenté
cent
mélanger
équipe
fil
coût
perdu
brun
porter
jardin
égal
expédié
choisir
est tombé
s’adapter
débit
juste
banque
recueillir
sauver
contrôle
décimal
oreille
autre
tout à fait
cassé
cas
milieu
tuer
fils
lac
moment
échelle
fort
printemps
observer
enfant
droit
consonne
nation
dictionnaire
lait
vitesse
méthode
organe
payer
âge
section
robe
nuage
surprise
calme
pierre
minuscule
montée
frais
conception
pauvres
lot
expérience
bas
clé
fer
unique
bâton
plat
vingt
peau
sourire
pli
trou
sauter
bébé
huit
village
se rencontrent
racine
acheter
augmenter
résoudre
métal
si
pousser
sept
paragraphe
troisième
doit
en attente
cheveux
décrire
cuisinier
étage
chaque
résultat
brûler
colline
coffre-fort
chat
siècle
envisager
type
droit
peu
côte
copie
phrase
silencieux
haut
sable
sol
rouleau
température
doigt
industrie
valeur
lutte
mensonge
battre
exciter
naturel
vue
sens
capital
ne sera pas
chaise
danger
fruit
riche
épais
soldat
processus
fonctionner
pratique
séparé
difficile
médecin
s’il vous plaît
protéger
midi
récolte
moderne
élément
frapper
étudiant
coin
partie
alimentation
dont
localiser
anneau
caractère
insecte
pris
période
indiquer
radio
rayon
atome
humain
histoire
effet
électrique
attendre
os
rail
imaginer
fournir
se mettre d’accord
ainsi
doux
femme
capitaine
deviner
nécessaire
net
aile
créer
voisin
lavage
chauve-souris
plutôt
foule
blé
comparer
poème
chaîne
cloche
dépendre
viande
rub
tube
célèbre
dollar
courant
peur
vue
mince
triangle
planète
se dépêcher
chef
colonie
horloge
mine
lien
entrer
majeur
frais
recherche
envoyer
jaune
pistolet
permettre
impression
mort
place
désert
costume
courant
ascenseur
rose
arriver
maître
piste
mère
rivage
division
feuille
substance
favoriser
relier
poste
passer
corde
graisse
heureux
original
part
station
papa
pain
charger
propre
bar
proposition
segment
esclave
canard
instant
marché
degré
peupler
poussin
cher
ennemi
répondre
boisson
se produire
support
discours
nature
gamme
vapeur
mouvement
chemin
liquide
enregistrer
signifiait
quotient
dents
coquille
cou
oxygène
sucre
décès
assez
compétence
femmes
saison
solution
aimant
argent
merci
branche
rencontre
suffixe
particulièrement
figue
peur
énorme
sœur
acier
discuter
avant
similaire
guider
expérience
score
pomme
acheté
LED
pas
manteau
masse
carte
bande
corde
glissement
gagner
rêver
soirée
condition
alimentation
outil
total
de base
odeur
vallée
ni
double
siège
continuer
bloc
graphique
chapeau
vendre
succès
entreprise
soustraire
événement
particulier
accord
baignade
terme
opposé
femme
chaussure
épaule
propagation
organiser
camp
inventer
coton
né
déterminer
litre
neuf
camion
bruit
niveau
chance
recueillir
boutique
tronçon
jeter
éclat
propriété
colonne
molécule
sélectionner
mal
gris
répétition
exiger
large
préparer
sel
nez
pluriel
colère
revendication
continent
"""
        self.process_common_words()
        return

    def get_min_threshold_intersection_pct(
            self
    ):
        return French.MIN_FR_SENT_INTERSECTION_PCT

if __name__ == '__main__':
    obj = French()
    print(obj.common_words)
    exit(0)
