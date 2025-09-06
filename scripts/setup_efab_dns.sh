#!/bin/bash

# Script to setup eFab DNS in WSL
# Run this if eFab works from Windows but not WSL

echo "eFab DNS Setup for WSL"
echo "======================"
echo ""

# Check if we can resolve from Windows
echo "Checking Windows DNS resolution..."
WINDOWS_IP=$(powershell.exe -Command "(Resolve-DnsName efab.bklapps.com -ErrorAction SilentlyContinue).IPAddress" 2>/dev/null | tr -d '\r')

if [ -z "$WINDOWS_IP" ]; then
    echo "❌ Could not resolve efab.bklapps.com from Windows"
    echo ""
    echo "Please ensure:"
    echo "1. You're connected to the corporate network/VPN"
    echo "2. The domain is accessible from Windows"
    echo ""
    echo "You can manually add the IP if you know it:"
    echo "  sudo bash -c 'echo \"IP_ADDRESS efab.bklapps.com\" >> /etc/hosts'"
    exit 1
fi

echo "✅ Windows resolved efab.bklapps.com to: $WINDOWS_IP"
echo ""

# Check if already in hosts file
if grep -q "efab.bklapps.com" /etc/hosts; then
    echo "⚠️  efab.bklapps.com already in /etc/hosts:"
    grep "efab.bklapps.com" /etc/hosts
    echo ""
    read -p "Do you want to update it? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Remove old entry
        sudo sed -i '/efab.bklapps.com/d' /etc/hosts
    else
        echo "Keeping existing entry."
        exit 0
    fi
fi

# Add to hosts file
echo "Adding to /etc/hosts..."
echo "$WINDOWS_IP efab.bklapps.com" | sudo tee -a /etc/hosts > /dev/null

if [ $? -eq 0 ]; then
    echo "✅ Successfully added to /etc/hosts"
    echo ""
    
    # Test the connection
    echo "Testing connection..."
    if ping -c 1 efab.bklapps.com > /dev/null 2>&1; then
        echo "✅ Can now reach efab.bklapps.com!"
        echo ""
        
        # Test with Python
        echo "Testing with Python..."
        python3 -c "import socket; print(f'✅ Python resolved: {socket.gethostbyname(\"efab.bklapps.com\")}')"
        
        echo ""
        echo "✨ Setup complete! You can now use the eFab API from WSL."
    else
        echo "⚠️  Added to hosts but still can't ping. There may be a firewall issue."
    fi
else
    echo "❌ Failed to update /etc/hosts. Try running with sudo:"
    echo "  sudo bash setup_efab_dns.sh"
fi