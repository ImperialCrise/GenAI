=Diffusion en temps discret (DDPM, DDIM uniquement)

==DDPM minimal sur MNIST + trajectoires de débruitage

===Implémenter un DDPM (prédiction du bruit (\epsilon)) sur MNIST. Visualiser le forward noising et le reverse denoising étape par étape, puis générer des échantillons. Tester 1–2 schedules simples (linéaire) et montrer l’effet du nombre de pas sur la qualité.

Objectifs:

- Implémenter processus avant (q(x_t|x_{t-1})) et perte “simple”.

- Construire un petit U-Net (ou CNN) pour (\epsilon_\theta(x_t,t)).

- Générer et visualiser les trajectoires de débruitage.

- Faire une évaluation simple (FID léger ou score via classifieur).

Références:

- Ho, Jain, Abbeel — Denoising Diffusion Probabilistic Models (arXiv:2006.11239). (arXiv)

