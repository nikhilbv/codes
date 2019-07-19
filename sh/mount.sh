## Mounting

**Mounting Remote File System over ssh**
* How To Use SSHFS to Mount Remote File Systems Over SSH
* https://www.digitalocean.com/community/tutorials/how-to-use-sshfs-to-mount-remote-file-systems-over-ssh

sudo apt install sshfs
sudo mkdir /mnt/droplet <--replace "droplet" whatever you prefer
sudo sshfs -o allow_other,default_permissions <userName>@<IP>:</remote/path> </local/mount/path>
sudo umount /mnt/droplet


sudo sshfs -o allow_other,default_permissions,IdentityFile=~/.ssh/id_rsa root@xxx.xxx.xxx.xxx:/ /mnt/droplet
