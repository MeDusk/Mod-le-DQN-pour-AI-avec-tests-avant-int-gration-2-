
## Introduction

Dans le cadre du projet Robot Puissance 4, j'ai été chargé de développer la partie Intelligence Artificielle. L'objectif était de créer un agent capable de jouer au Puissance 4 de manière autonome contre un joueur humain. L'agent sera ensuite intégré dans un robot physique qui pourra exécuter les coups sur un vrai plateau.

**Projet** : Smart-Connect4 - Robot joueur de Puissance 4  

***

## Choix de l'Approche

### Pourquoi le Deep Q-Learning ?

Au début, j'ai hésité entre plusieurs approches. Je pouvais utiliser :
- Un algorithme Minimax avec élagage Alpha-Beta (approche classique)
- Monte Carlo Tree Search (MCTS)
- Deep Reinforcement Learning

J'ai finalement choisi le **Deep Q-Network (DQN)** parce que :
- L'agent apprend tout seul sans qu'on ait besoin de programmer les stratégies
- Il peut s'améliorer au fil du temps
- C'est plus intéressant du point de vue apprentissage machine
- Ça correspond bien à ce qu'on a étudié en cours de Deep RL

### Architecture Dueling DQN

J'ai implémenté une version améliorée du DQN classique : le **Dueling DQN**. 

L'idée c'est de séparer le réseau en deux parties :
- Une partie qui estime **la valeur d'un état** V(s) : "Est-ce que je suis dans une bonne situation ?"
- Une partie qui estime **l'avantage de chaque action** A(s,a) : "Quelle action est meilleure que les autres ?"

À la fin, on combine les deux pour obtenir les Q-values : **Q(s,a) = V(s) + A(s,a) - moyenne(A)**

Cette architecture marche mieux parce que parfois l'état est tellement mauvais (ou bon) que peu importe l'action qu'on choisit, ça ne change pas grand chose. Le Dueling DQN arrive mieux à détecter ces situations.

***

## Implémentation

### Structure du Code

J'ai organisé mon code en plusieurs fichiers pour que ce soit plus propre :

**dqn_model.py** : L'architecture du réseau
- 3 couches convolutionnelles (64, 128, 128 filtres)
- BatchNormalization pour stabiliser
- Séparation en deux streams (Value et Advantage)

**dqn_agent.py** : La logique de l'agent
- Gestion de la mémoire de replay
- Stratégie epsilon-greedy (exploration vs exploitation)
- Mise à jour du réseau
- Sauvegarde/chargement des modèles

**connect_four_env.py** : L'environnement de jeu
- Simulation du plateau de Puissance 4
- Règles du jeu
- Calcul des récompenses
- Détection des victoires

**train_dqn.py** : Script d'entraînement
- Boucle d'entraînement principale
- Self-play (l'agent joue contre lui-même)
- Suivi des performances

### Représentation de l'État

Pour que le réseau de neurones puisse comprendre le plateau, je l'ai représenté comme un tenseur 3D de taille (3, 6, 7) :
- **Canal 0** : Mes pions (1 si présent, 0 sinon)
- **Canal 1** : Les pions adverses
- **Canal 2** : Les cases vides

Comme ça, le réseau convolutionnel peut détecter des patterns comme les alignements de 3 pions, etc.

### Techniques Utilisées

**1. Prioritized Experience Replay**

Au lieu de prendre des expériences au hasard dans la mémoire, je donne plus d'importance aux expériences où l'agent s'est trompé (erreur TD élevée). Ça permet d'apprendre plus vite parce qu'on se concentre sur les situations difficiles.

**2. Double DQN**

Un problème classique du DQN c'est qu'il surestime les Q-values. Pour corriger ça, j'utilise :
- Le réseau principal pour **choisir** la meilleure action
- Le réseau cible pour **évaluer** cette action

Ça donne des estimations plus réalistes.

**3. Gradient Clipping**

Pour éviter que les gradients explosent pendant l'entraînement, je les limite à une norme maximale de 10. Ça stabilise beaucoup l'apprentissage.

**4. Masquage des Actions Invalides**

Pendant le jeu, certaines colonnes sont pleines. Pour éviter que l'agent essaye de jouer dedans, je masque ces actions en leur donnant une Q-value de -∞. Comme ça, l'agent choisit toujours un coup légal.

***

## Entraînement

### Hyperparamètres

Après plusieurs tests, j'ai utilisé ces paramètres :

```
Learning Rate : 0.0001
Gamma (γ) : 0.99
Batch Size : 64
Taille du Buffer : 100 000
Target Update : tous les 500 épisodes
Epsilon Start : 1.0
Epsilon End : 0.01
Epsilon Decay : 20 000 steps
```

### Self-Play

L'agent s'entraîne en jouant contre lui-même. Au début j'ai testé contre un adversaire aléatoire, mais ça ne marchait pas bien parce que :
- L'agent apprenait à battre un joueur nul
- Une fois qu'il gagnait tout le temps, il n'apprenait plus rien

Avec le self-play :
- L'adversaire s'améliore en même temps que l'agent
- Le niveau de difficulté augmente progressivement
- L'agent ne peut pas tricher ou exploiter les faiblesses d'un adversaire fixe

J'alterne qui commence à chaque épisode pour que l'agent apprenne à jouer dans les deux positions.

### Récompenses

J'ai utilisé un système de récompenses simple :
- **+100** si l'agent gagne
- **-100** si l'agent perd
- **0** pour un match nul
- **0** pour tous les coups intermédiaires

Au début j'avais essayé de donner des petites récompenses pour les bons coups (créer un alignement de 2, bloquer l'adversaire, etc.) mais ça compliquait tout et ça marchait moins bien. Le système simple fonctionne mieux.

***

## Résultats

### Courbe d'Apprentissage

- **0-2000 épisodes** : L'agent explore beaucoup, il joue presque aléatoirement. La récompense est très négative (autour de -100).

- **2000-6000 épisodes** : Phase d'apprentissage rapide. L'agent commence à comprendre les bases : ne pas laisser l'adversaire aligner 4 pions, essayer de créer ses propres alignements.

- **6000-10000 épisodes** : L'agent développe des vraies stratégies. Il apprend à privilégier le centre, à créer des menaces multiples. La récompense devient positive.

- **10000-15000 épisodes** : Convergence. L'agent atteint un plateau autour de +40 à +50 de récompense moyenne. Il y a toujours de la variance parce qu'en self-play, les deux joueurs ont le même niveau.

### Performance Finale

Après 15 000 épisodes, l'agent :
- Gagne environ 55-60% des parties contre lui-même
- Bloque systématiquement quand l'adversaire a 3 pions alignés
- Crée activement des situations où il a deux façons de gagner
- Préfère jouer au centre en début de partie (stratégie optimale)
- Ne joue plus jamais de coups illégaux

***

## Difficultés Rencontrées

### 1. Variance en Self-Play

Quand l'agent joue contre lui-même, les performances fluctuent beaucoup. C'est normal parce que si les deux joueurs sont de niveau égal, le win rate devrait être autour de 50%. Pour suivre la vraie progression, j'ai dû lisser la courbe sur 100 épisodes.

### 2. Temps d'Entraînement

15 000 épisodes ça prend du temps ! Sur CPU c'était beaucoup trop long (plusieurs jours). J'ai dû :
- Utiliser un GPU (CUDA) pour accélérer
- Optimiser le code (vectorisation avec NumPy)
- Lancer l'entraînement overnight

Au final ça a pris environ 6-8 heures sur GPU.

### 3. Tuning des Hyperparamètres

Trouver les bons hyperparamètres a demandé pas mal d'essais. Par exemple :
- Si epsilon decay trop vite → l'agent n'explore pas assez et se bloque dans une stratégie sous-optimale
- Si learning rate trop élevé → instabilité
- Si target update trop fréquent → le réseau cible change trop vite et l'apprentissage diverge

J'ai dû faire plusieurs runs pour trouver le bon équilibre.



***

**Auteur** : Mohamed NAJID  
**Projet** : Robot puissance 4 - M2 IA UCBL  
