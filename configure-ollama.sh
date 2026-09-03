#!/bin/bash
# Script to configure Ollama to bind to 0.0.0.0:11434

echo "Configuring Ollama to bind to 0.0.0.0:11434..."

# Create systemd override directory
sudo mkdir -p /etc/systemd/system/ollama.service.d

# Create override file
sudo tee /etc/systemd/system/ollama.service.d/override.conf > /dev/null <<EOF
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
EOF

# Reload systemd configuration
sudo systemctl daemon-reload

# Restart Ollama service
sudo systemctl restart ollama

# Wait a moment for the service to start
sleep 2

# Check service status
echo ""
echo "Checking Ollama service status..."
sudo systemctl status ollama --no-pager -l | head -15

# Verify it's listening on 0.0.0.0:11434
echo ""
echo "Checking listening ports..."
sudo ss -tlnp | grep 11434 || sudo netstat -tlnp | grep 11434

echo ""
echo "Configuration complete!"
echo "Ollama should now be bound to 0.0.0.0:11434"
