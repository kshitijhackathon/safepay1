# Base Node.js image
FROM node:20-slim AS node_base

# Set working directory
WORKDIR /app

# Install Python and required system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Create Python virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy package files
COPY package*.json ./

# Install Node.js dependencies
RUN npm ci --only=production

# Copy Python requirements
COPY pyproject.toml ./

# Install Python dependencies
COPY . .
RUN pip3 install -e .

# Build the application
RUN npm run build

# Use a smaller base image for production
FROM node:20-slim

WORKDIR /app

# Copy necessary files from build stage
COPY --from=node_base /app/dist ./dist
COPY --from=node_base /app/node_modules ./node_modules
COPY --from=node_base /opt/venv /opt/venv
COPY --from=node_base /app/server ./server
COPY --from=node_base /app/models ./models
COPY --from=node_base /app/migrations ./migrations

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH"
ENV NODE_ENV=production

# Expose ports
EXPOSE 5000

# Start the application
CMD ["npm", "start"]