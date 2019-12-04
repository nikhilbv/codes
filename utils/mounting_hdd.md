## Mounting the HDD for AI development machines in Ubuntu 18.04 
1. Show Applications on bottom left -> Disks
2. Select the perticular hark disk from the list
3. First unmount the HDD if it is already mounted by clicking on stop button
4. Click on settings button -> Edit Mount Options
  * Off the User Session Defaults
  * Check the Mount at system startup
  * Check  the Show in user interface
  * Set the Mount Point to `/mnt/sdb1`
5. Create a `aimldl-dat` folder in `/mnt/sdb1`
```bash
cd /mnt/sdb1
mkdir aimldl-dat
```
6. Assign the user permissions 
```bash
cd /mnt/sdb1
sudo chown -R $(id -un):$(id -gn) aimldl-dat
```
7. Delete the `aimldl-dat` at `/` level using sudo 
```bash
cd /
sudo rm -rf aimldl-dat
```
8. Symbolically link the `/mnt/sdb1/aimldl-dat` at `/` level using sudo 
```bash
cd /
sudo ln -s /mnt/sdb1/aimldl-dat 
```

## Changing to static ip
1. Settings -> Network
2. Select IPv4
3. Change IPv4 Method from Automatic (DHCP) to Manual
4. Addresses
  * Set Address to the system IP (ifconfig) ex- 10.4.71.59
  * Set Netmask to 255.255.255.0
  * Set Gateway to 10.4.71.1
5. DNS
  * Set DNS to 10.4.71.1