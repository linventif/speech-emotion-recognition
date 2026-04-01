# Présentation Master

## Speech Emotion Recognition
### Comment fonctionne le projet, et en quoi il s'inspire du papier de recherche

Support préparé à partir de :
- l'article `information-16-00518.pdf` : *Speech Emotion Recognition on MELD and RAVDESS Datasets Using CNN* (Waleed, Shaker, 2025)
- le notebook du projet [`CNN.ipynb`](/home/linventif/speech-emotion-recognition/CNN.ipynb)
- le modèle entraîné [`models/cnn.keras`](/home/linventif/speech-emotion-recognition/models/cnn.keras)
- la démo locale [`demo_app.py`](/home/linventif/speech-emotion-recognition/demo_app.py) et [`demo/index.html`](/home/linventif/speech-emotion-recognition/demo/index.html)

---

## Slide 1 - Titre et idée générale

**Titre**
Speech Emotion Recognition avec CNN 1D sur audio vocal

**Message clé**
L'objectif du projet est de prédire l'émotion portée par une voix à partir d'un signal audio, sans utiliser le texte ni la vidéo.

**À dire à l'oral**
- Le problème traité est la reconnaissance automatique des émotions dans la parole.
- On part d'un signal audio brut, on en extrait des caractéristiques acoustiques, puis on les donne à un réseau de neurones convolutif.
- L'intérêt est d'améliorer l'interaction humain-machine : assistants vocaux, santé, support client, robotique sociale.

---

## Slide 2 - Pourquoi ce sujet est intéressant

**Contexte**
- La voix ne transmet pas seulement des mots.
- Elle transmet aussi de l'intonation, de l'énergie, du rythme et des indices émotionnels.
- En pratique, reconnaître la colère, la joie, la peur ou la tristesse peut enrichir un système conversationnel.

**Lien avec le papier**
L'article insiste sur trois contraintes :
- garder l'information temporelle du signal
- rester léger en calcul
- viser un usage en temps réel

**À dire à l'oral**
- Beaucoup de travaux transforment l'audio en image de spectrogramme puis utilisent des CNN 2D.
- Le papier défend une autre approche : utiliser des features séquentielles et un CNN 1D plus léger.

---

## Slide 3 - Question de recherche

**Question**
Peut-on reconnaître les émotions de la parole avec un modèle suffisamment précis, léger et exploitable en quasi temps réel ?

**Hypothèse du papier**
La fusion de plusieurs familles de features acoustiques donne de meilleurs résultats qu'une seule feature.

**Features utilisées**
- MFCC : résumé spectral proche de la perception humaine
- Mel-spectrogram : répartition de l'énergie dans les bandes fréquentielles
- Chroma : répartition de l'énergie sur 12 classes de hauteur musicale

**À dire à l'oral**
- L'idée centrale est que ces trois représentations sont complémentaires.
- MFCC capte bien le timbre.
- Mel-spectrogram capte la structure fréquentielle.
- Chroma apporte une information harmonique supplémentaire.

---

## Slide 4 - Dataset utilisé dans ce repo

**Dataset**
RAVDESS

**Ce qu'il contient**
- 24 acteurs
- énoncés joués
- 8 émotions
- fichiers audio `.wav`

**Émotions du projet**
- neutral
- calm
- happy
- sad
- angry
- fearful
- disgust
- surprised

**Implémentation dans le repo**
Le mapping des émotions est défini dans [`CNN.ipynb`](/home/linventif/speech-emotion-recognition/CNN.ipynb) avec les codes `01` à `08`.

**À dire à l'oral**
- Le repo ne reprend pas MELD.
- Il se concentre sur RAVDESS, qui est plus propre et plus contrôlé.
- C'est bien pour entraîner un premier modèle, mais moins représentatif de conversations réelles bruyantes.

---

## Slide 5 - Pipeline global du projet

**Pipeline**
1. Charger un fichier audio
2. Extraire des features acoustiques
3. Augmenter les données pour améliorer la robustesse
4. Encoder les labels d'émotion
5. Entraîner un CNN 1D
6. Évaluer avec accuracy, matrice de confusion et F1-score
7. Faire l'inférence sur un nouvel enregistrement

**À dire à l'oral**
- Le pipeline est classique en apprentissage supervisé.
- La partie importante n'est pas seulement le réseau.
- La qualité des features et du prétraitement joue un rôle majeur.

---

## Slide 6 - Prétraitement et data augmentation

**Dans le notebook**
Le chargement des données et l'augmentation sont faits dans [`CNN.ipynb`](/home/linventif/speech-emotion-recognition/CNN.ipynb).

**Deux augmentations**
- ajout de bruit gaussien
- décalage temporel du signal

**Pourquoi**
- éviter le surapprentissage
- rendre le modèle moins sensible aux petites variations du signal

**À dire à l'oral**
- Pour chaque fichier audio, le notebook crée plusieurs variantes.
- Cela enrichit artificiellement le dataset.
- C'est cohérent avec le papier, qui dit que bruit et shifting améliorent la généralisation.

---

## Slide 7 - Extraction des features

**Fonction clé**
`extract_feature(data, sr, mfcc=True, chroma=True, mel=True)`

**Dimensions obtenues**
- 40 coefficients MFCC
- 12 coefficients Chroma
- 128 coefficients Mel
- total : 180 features

**Important**
Dans ce repo, on fait une moyenne temporelle des représentations pour obtenir un vecteur final de taille 180.

**À dire à l'oral**
- C'est un point important.
- Le papier parle d'une architecture multi-branches qui traite des séquences de features.
- Ici, l'implémentation concatène des résumés statistiques moyens dans un seul vecteur.
- Donc le repo suit l'esprit du papier, mais pas exactement son architecture complète.

---

## Slide 8 - Architecture du modèle dans le repo

**Entrée**
- tenseur `(180, 1)`

**Architecture**
- Conv1D 256 filtres, kernel 5
- Conv1D 128 filtres, kernel 5
- Dropout 0.1
- MaxPooling1D
- Conv1D 128 filtres
- Conv1D 128 filtres
- Dropout 0.5
- Flatten
- Dense 8
- Softmax

**Source**
Défini dans [`CNN.ipynb`](/home/linventif/speech-emotion-recognition/CNN.ipynb), modèle sauvegardé dans [`models/cnn.keras`](/home/linventif/speech-emotion-recognition/models/cnn.keras).

**Paramètres**
Le modèle chargé affiche environ 352k paramètres entraînables et 1.056M paramètres au total avec l'état de l'optimiseur.

**À dire à l'oral**
- Le réseau apprend des motifs locaux dans le vecteur de features grâce aux convolutions 1D.
- La couche finale produit une probabilité pour chacune des 8 émotions.

---

## Slide 9 - Différence entre le papier et ce repo

**Ce que propose le papier**
- deux datasets : MELD et RAVDESS
- fusion multi-features
- branches 1D-CNN parallèles dédiées à chaque famille de features
- objectif de modèle léger pour usage temps réel

**Ce que fait réellement ce repo**
- un seul dataset : RAVDESS
- features MFCC + Chroma + Mel
- concaténation en un vecteur unique de taille 180
- un CNN 1D séquentiel simple, pas une vraie fusion parallèle multi-branches

**Conclusion honnête**
Le repo est cohérent avec l'idée scientifique du papier, mais c'est une version simplifiée de cette idée.

**À dire à l'oral**
- C'est important de le dire devant un jury ou un groupe de Master.
- Sinon on risque de survendre l'implémentation.
- Le projet est inspiré par la logique de fusion de features, mais n'implémente pas toute la sophistication de l'article.

---

## Slide 10 - Entraînement et métriques

**Dans le notebook**
- split train/test : `test_size=0.25`
- labels encodés avec `LabelEncoder`
- optimiseur Adam avec `learning_rate=1e-3`
- entraînement sur 100 epochs
- batch size 64

**Évaluation visible dans le notebook**
- accuracy de validation finale autour de `0.9093`
- F1-score autour de `0.9095`

**À dire à l'oral**
- Les métriques du notebook montrent un bon niveau de performance sur RAVDESS.
- Mais il faut rappeler que RAVDESS est un dataset propre, joué par des acteurs, donc plus facile que la vraie vie.

---

## Slide 11 - Résultats du papier

**Selon l'article**
- accuracy sur RAVDESS : `91.9%`
- accuracy sur MELD : `94.0%`
- étude d'ablation :
  - MFCC seul : `85.0%` sur RAVDESS
  - MFCC + Mel : `89.5%`
  - MFCC + Mel + Chroma : `91.9%`

**Ce que cela montre**
- la fusion des features améliore clairement les performances
- les features sont complémentaires

**Point critique**
La conclusion du papier inverse une fois les scores MELD/RAVDESS.
Le reste du papier indique bien `91.9%` pour RAVDESS et `94.0%` pour MELD.

**À dire à l'oral**
- Cette remarque montre une lecture critique.
- Ce n'est pas grave scientifiquement, mais c'est une incohérence éditoriale à signaler.

---

## Slide 12 - Démo live du projet

**Ce qui a été ajouté**
- un backend local : [`demo_app.py`](/home/linventif/speech-emotion-recognition/demo_app.py)
- une interface web d'enregistrement : [`demo/index.html`](/home/linventif/speech-emotion-recognition/demo/index.html)

**Fonctionnement**
1. l'utilisateur enregistre sa voix
2. le navigateur convertit l'audio en WAV
3. le serveur sauvegarde le fichier dans `recordings/`
4. le modèle chargé depuis `models/cnn.keras` fait la prédiction
5. la page affiche l'émotion et les scores

**À dire à l'oral**
- Cela permet de transformer un notebook en démonstrateur concret.
- On passe d'un projet de recherche à un prototype d'usage.

---

## Slide 13 - Forces du projet

**Forces**
- pipeline clair et reproductible
- modèle relativement léger
- bonnes performances sur RAVDESS
- démo locale simple à comprendre
- lien cohérent avec un papier récent

**À dire à l'oral**
- Pour un projet académique, c'est une bonne base.
- Le projet est assez simple pour être expliqué, mais assez concret pour être démontré.

---

## Slide 14 - Limites

**Limites techniques**
- dataset joué, pas spontané
- peu robuste au bruit réel
- pas de texte, pas de vidéo, pas de contexte conversationnel
- implémentation plus simple que l'architecture du papier
- moyenne temporelle des features, donc perte d'information dynamique fine

**À dire à l'oral**
- Le principal risque est d'obtenir d'excellents résultats sur dataset propre et de moins bien fonctionner sur une vraie conversation.
- C'est une limite très classique en speech emotion recognition.

---

## Slide 15 - Pistes d'amélioration

**Améliorations possibles**
- implémenter la vraie architecture multi-branches du papier
- conserver des séquences temporelles au lieu de moyenner les features
- ajouter MELD, IEMOCAP ou CREMA-D pour tester la généralisation
- faire une vraie détection en streaming temps réel
- comparer avec Wav2Vec2 ou HuBERT
- ajouter calibration et explication des prédictions

**À dire à l'oral**
- La prochaine vraie marche scientifique serait de comparer l'implémentation actuelle avec une version plus fidèle au papier.

---

## Slide 16 - Conclusion

**Conclusion**
- Le projet cherche à reconnaître l'émotion dans la voix.
- Il utilise des features acoustiques classiques mais efficaces.
- Il les donne à un CNN 1D entraîné sur RAVDESS.
- Les résultats sont solides sur ce dataset.
- La démo montre qu'on peut déjà en faire un prototype interactif.
- Mais l'implémentation actuelle reste une simplification du papier de recherche.

**Phrase de fin possible**
Ce projet montre bien comment on peut passer d'une intuition scientifique, la fusion de features acoustiques, à un prototype fonctionnel, tout en gardant un regard critique sur les limites de généralisation.

---

## Version courte pour 5 minutes

Si tu dois aller vite, garde seulement :
- Slide 1 : sujet et objectif
- Slide 4 : dataset RAVDESS
- Slide 7 : features MFCC, Mel, Chroma
- Slide 8 : CNN 1D
- Slide 9 : différence papier vs repo
- Slide 11 : résultats
- Slide 12 : démo live
- Slide 16 : conclusion

---

## Questions probables du groupe

**Pourquoi un CNN 1D et pas un CNN 2D ?**
- Parce qu'on traite une séquence de features plutôt qu'une image.
- C'est plus léger en calcul.

**Pourquoi RAVDESS est-il insuffisant seul ?**
- Parce que c'est un corpus joué, propre et contrôlé.
- La vraie parole est plus bruitée et ambiguë.

**Pourquoi la fusion de features aide ?**
- Parce que chaque feature capture une dimension différente du signal.

**Le projet implémente-t-il exactement le papier ?**
- Non.
- Il en reprend l'idée générale, mais dans une forme simplifiée.

**Peut-on faire du temps réel ?**
- Oui en mode quasi temps réel avec la démo actuelle.
- Pour du vrai streaming, il faudrait découper l'audio en fenêtres et prédire en continu.
