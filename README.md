# Deep learning BBox + Class detection

Création d'un réseau de Deep Learning permettant d'analyser une image, de reconnaitre un objet étudié dans l'image (parmi une multitude de classes) et de tracer un rectangle afin de l'encadrer. En clair, le réseau permet d'identifier la position d'un objet, et de déterminer sa classe suite à un entrainement de type supervisé.

NB : On ne peut détecter une seule instance de l'objet dans l'image.

## Exemple de classification utilisé

L'exemple actuel utilise un dataset contenant des avions et des motos. Ainsi, suite à un entrainement, l'intelligence artificielle est capable de détecter si l'image possède un avion ou une moto, et est capable de déterminer la zone dans lequel cet objet se situe.

## Système générique

Le système a été créé dans l'objectif d'être générique, et ainsi de pouvoir fonctionner avec différents datasets. Il est ainsi possible de prédire autre chose que des emplacements d'avion... Afin de tester avec un dataset personnalisé, il suffit juste de déployer le datasets concerné dans les répertoires suivants :

> `datasets/*/Images` pour les images contenant l'objet à détecter ;

> `datasets/*/Annotations` pour les annotations, contenant la position (top_left_x, top__left_y, bottom_right_x, bottom_right_y) de l'objet dans l'image ;