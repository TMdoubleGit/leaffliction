# leaffliction


## Taille dataset :

apple_black_rot     620     ->      1640
apple_healthy        1640  
apple_rust          275     ->      1640
apple_scab          629     ->      1640

grape_black_rot     1178     ->      1640
grape_esca          1382     ->      1640
grape_healthy       422      ->      1640
grape_spot          1075     ->      1640

## Question ?

Pour augmentation.py, est ce que les dataset doivent etre parfaitement equilibre ?
Est ce qu'il faut equilibrees les apples et les grapes par rapport au sous dossier qui a le + d'images OU il les equilibres par rapport a leur categorie 
(nb images max d'apple pour les apples et nb images max de grapes pour les grapes) ?
SOLUTION : 1640 pour tous

Pour transformation.py, est ce que l'extraction des characteristiques a vraiment une utilite pour la correction ?
Est ce que la transformation est faite pour toutes les images originales + celles augmentees ?
SOLUTION : OUI FORMAT UNE IMAGE POUR LES 1 + 5 IMAGES TRANSFORMEES


## TODO :
- changer crop dans augmentation (pb de taille) ✅
- probleme avec contrast ✅
- secure les dump et save model ✅
- faire la partie archivahe ZIP ✅
- partie predict display image ✅
- recup final_directory

pour le predict : besoin d'une fonction par transformation
pour le train : besoin d'une fonction de transformation de dataset


## A retourner dans le .zip : 

- augmented_directory (part2)
- dataset 400 train / 100 validation (part 4)
- saved_model (part4)
- increased/modified images (part4)