# Cours — Simulation photonique (FDTD/EME/FEM) et interaction fonderie

Document maître unique, compilé en XeLaTeX.

## Structure

```
cours/
  cours_fdtd_eme_fem.tex   <- document maître : préambule, page de garde,
                               table des matières, \include des modules
  style_cours.sty          <- palette, boîtes pédagogiques (Objectifs,
                               Prérequis, Rappel, Lien fonderie, Livrable,
                               Ressources, Exercice, Diagnostic), style de
                               code — source unique de vérité visuelle
  modules/
    module0_rappels.tex      Module 0 — Rappels fondamentaux (4 chapitres)
    module1_fdtd.tex         Module 1 — FDTD open source / Meep (5 chapitres)
    module2_eme.tex          Module 2 — EME (3 chapitres)
    module3_fem.tex          Module 3 — FEM (4 chapitres)
    module4_lumerical.tex    Module 4 — Ansys Lumerical pour le PIC design (8 chapitres)
    module5_layout_pdk.tex   Module 5 — Layout et PDK (5 chapitres)
    module6_fonderie.tex     Module 6 — Interaction avec la fonderie (4 chapitres)
    module7_projet.tex       Module 7 — Projet de synthèse (3 chapitres)
```

## État actuel

**PLAN pour l'ensemble du cours, à l'exception du Chapitre 0.1 (COMPLET).**
Chaque chapitre présente son titre, sa position dans le cours, ses objectifs
pédagogiques, ses prérequis, ses outils, la nécessité (ou non) d'un encart
de rappel/mise à niveau, un aperçu du contenu prévu et son livrable prévu.
Chaque chapitre porte l'annotation `\etatunite{PLAN}` (ou `COMPLET`) en
tête ; à retirer manuellement une fois tout le cours passé en contenu
complet.

Le Chapitre 0.1 (« Optique guidée et modes ») dans
`modules/module0_rappels.tex` a été rédigé intégralement (rappel, théorie,
équation de dispersion TE, script Python de résolution + profil de mode,
lien fonderie, 2 exercices + 1 diagnostic, livrable, ressources) comme
exemple de référence pour calibrer le niveau de détail, la longueur et le
style attendus des chapitres suivants.

Le reste du plan est destiné à être validé (numérotation des chapitres,
portée de chaque unité, outils retenus — en particulier le choix du
solveur EME et du solveur FEM open source pour les Modules 2 et 3, encore
à trancher) avant génération du contenu pédagogique complet des chapitres
restants, un par un.

## Compilation

Deux passes XeLaTeX (trois si la table des matières ou des références
changent) :

```bash
cd cours
xelatex cours_fdtd_eme_fem.tex
xelatex cours_fdtd_eme_fem.tex
```

Dépendances LaTeX : `fontspec`, `polyglossia`, `geometry`, `fancyhdr`,
`titlesec`, `graphicx`, `amsmath`/`amssymb`/`amsthm`, `siunitx`, `booktabs`,
`longtable`, `xcolor` (option `table`), `tcolorbox` (option `most`),
`hyperref`, `cleveref`, `enumitem`, `listings` — toutes disponibles dans une
distribution TeX Live standard (`texlive-latex-extra` + `texlive-xetex` +
`texlive-lang-french` couvrent l'ensemble).

Ce fichier n'a pas encore été compilé dans cet environnement (aucune
distribution LaTeX installée ici) ; la cohérence des accolades, des
environnements `\begin`/`\end` et des références croisées `\cref` a été
vérifiée par script, mais une compilation XeLaTeX réelle reste à faire avant
de considérer le document comme définitivement validé.

## Prochaine étape

Valider ce plan (contenu, portée, outils EME/FEM à confirmer), puis générer
le contenu complet unité par unité en réutilisant le squelette détaillé du
prompt-cadre (objectifs → prérequis → outils → rappel → contenu théorique →
simulation guidée → lien fonderie → exercices → livrable → ressources), en
remplaçant chaque `\etatunite{PLAN}` par `\etatunite{COMPLET}` au fur et à
mesure.
