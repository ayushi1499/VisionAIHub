#!/bin/bash

# Create and activate virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate


# Install system dependencies
echo "Installing system dependencies..."
brew install ffmpeg

# Create packages.txt file if it doesn't exist
if [ ! -f packages.txt ]; then
    echo "Creating packages.txt file..."
    cat > packages.txt << EOL
ffmpeg
libsm6
libxext6
libgl1
EOL
    echo "packages.txt created successfully."
else
    echo "packages.txt already exists."
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Run the Streamlit app
echo "Starting Streamlit app..."
streamlit run app.py
