python3.9

TOCHANGE : bizarreries du code du paper
install a module : 
python -m pip install --user (module_name)

--resume_checkpoint ./weights/model000588.pt

nohup python sarddpm_train.py --save_interval 1 &
killall


python sarddpm_train.py --resume_checkpoint ./weights/model000588.pt --save_interval 1

en se connectant au GPU de ton choix
    jupyter-notebook --no-browser --port 8898

sur ton ordi perso: 
        ssh -J 1234567@ssh.ufr-info-p6.jussieu.fr -L 8898:localhost:8898 21112667@ppti-gpu-n

    Dans un navigateur copier ce que donne le notebook qui ressemble à ça:
        http://127.0.0.1:8898/?token=