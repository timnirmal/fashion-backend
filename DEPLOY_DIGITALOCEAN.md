## Deploying to DigitalOcean (Ubuntu 22.04/24.04)

This guide shows how to clone, build, run, and expose the FastAPI recommendation service on a DigitalOcean Droplet using Docker. It assumes a fresh Ubuntu VM with a non-root `sudo` user.

### 1) Provision a Droplet
- Create a new Ubuntu Droplet (>= 1 vCPU, 2GB RAM recommended)
- Add your SSH key; note the droplet public IP

### 2) SSH into the VM
```bash
ssh ubuntu@YOUR_DROPLET_IP
```

If using a different default user, update the username accordingly.

### 3) Install dependencies (Docker + Git)
```bash
sudo apt-get update -y
sudo apt-get install -y ca-certificates curl gnupg git

# Docker official repo
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo $VERSION_CODENAME) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update -y
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Optional: allow current user to run docker without sudo (re-login required)
sudo usermod -aG docker $USER
```

Logout and SSH back in for the docker group change to take effect.

### 4) Clone the repository
```bash
git clone YOUR_GIT_REMOTE_URL fashion-backend
cd fashion-backend
```

If the repo is private, ensure your SSH key has access.

### 5) Build the Docker image
```bash
docker build -t fashion-recsys:latest .
```

### 6) Run the container
Expose the API on port 8000. Optionally set allowed CORS origins for your frontend.

```bash
docker run -d --name fashion-recsys \
  -p 8000:8000 \
  -e ALLOW_ORIGINS="http://localhost:3000" \
  fashion-recsys:latest
```

Check logs if needed:
```bash
docker logs -f fashion-recsys
```

### 7) Open firewall ports (UFW)
On DigitalOcean Ubuntu images, `ufw` may be disabled by default. If enabled, allow port 8000:

```bash
sudo ufw allow OpenSSH
sudo ufw allow 8000/tcp
sudo ufw enable
sudo ufw status
```

Now your API should be reachable at `http://YOUR_DROPLET_IP:8000/health`.

### 8) Train and test the service
```bash
curl -X GET  http://YOUR_DROPLET_IP:8000/health
curl -X POST http://YOUR_DROPLET_IP:8000/train
curl -X POST http://YOUR_DROPLET_IP:8000/recommend -H 'Content-Type: application/json' \
  -d '{"user_id":"ffffffff-ffff-ffff-ffff-ffffffffffff","method":"blend_bpr_tfidf","top_k":10}'
```

### 9) Optional: Run as a systemd service
To ensure the container starts on reboot:

```bash
cat <<'UNIT' | sudo tee /etc/systemd/system/fashion-recsys.service
[Unit]
Description=Fashion Recsys API (Docker)
After=network.target docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/bin/docker run -d --name fashion-recsys -p 8000:8000 \
  -e ALLOW_ORIGINS="http://localhost:3000" fashion-recsys:latest
ExecStop=/usr/bin/docker rm -f fashion-recsys

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable fashion-recsys
sudo systemctl start fashion-recsys
```

### 10) Optional: Expose on 80/443 with Nginx (reverse proxy)
For a stable public URL and TLS, place Nginx in front of the container.

```bash
sudo apt-get install -y nginx
sudo tee /etc/nginx/sites-available/fashion-recsys >/dev/null <<'NGINX'
server {
    listen 80;
    server_name YOUR_DOMAIN_OR_IP;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
NGINX

sudo ln -s /etc/nginx/sites-available/fashion-recsys /etc/nginx/sites-enabled/fashion-recsys
sudo nginx -t && sudo systemctl reload nginx
```

Optionally obtain TLS certificates using Certbot:

```bash
sudo snap install --classic certbot
sudo ln -s /snap/bin/certbot /usr/bin/certbot
sudo certbot --nginx -d YOUR_DOMAIN
```

### 11) Maintenance
- Update image: `git pull && docker build -t fashion-recsys:latest . && sudo systemctl restart fashion-recsys`
- Logs: `docker logs -f fashion-recsys`
- Health: `curl http://YOUR_DOMAIN_OR_IP/health`


