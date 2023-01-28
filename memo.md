python3.9

TOCHANGE : bizarreries du code du paper
install a module : 
python -m pip install --user (module_name)

--resume_checkpoint ./weights/model000588.pt

nohup python sarddpm_train.py --resume_checkpoint ./weights/model000788.pt --save_interval 100 &
nohup python sarddpm_train.py --resume_checkpoint ./weights/model000599.pt --save_interval 10 &


nohup python sarddpm_train.py --save_interval 1 &

kill $(ps aux | grep 21112667 | awk '{print $2}' )
affiche
ps aux | grep 21112667

regarder tmux

all user x=system

python sarddpm_train.py --resume_checkpoint ./weights/model000588.pt --save_interval 1

en se connectant au GPU de ton choix
    jupyter-notebook --no-browser --port 8898

sur ton ordi perso: 
        ssh -J 1234567@ssh.ufr-info-p6.jussieu.fr -L 8898:localhost:8898 21112667@ppti-gpu-n

    Dans un navigateur copier ce que donne le notebook qui ressemble à ça:
        http://127.0.0.1:8898/?token=

Afficher l'historique git:
git log --oneline

Revenir à un commit:
git reset --HARD [token]

compresser:

tar -zcvf archive-file-name.tar.gz your-folder/
tar -zcvf testing.tar.gz testing/


Vf la taille d'un fichier
du -sh file

https://partage.imt.fr/index.php/s/nTi6eZEZ3NP3BPi