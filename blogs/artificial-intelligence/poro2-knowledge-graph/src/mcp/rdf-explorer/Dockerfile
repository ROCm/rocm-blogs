# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Multi-stage build for MCP RDF Explorer
# Stage 1: Build - Clone repository and download data
FROM python:3.12-slim AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN git clone https://github.com/emekaokoye/mcp-rdf-explorer.git

# Download MeSH SKOS data
RUN mkdir -p /build/data && \
    curl -L --retry 3 --retry-delay 2 \
    -o /build/data/mesh-skos.ttl \
    "https://finto.fi/rest/v1/mesh/data?format=text/turtle" \
    || { echo 'Failed to download MeSH SKOS data from finto.fi'; exit 1; } && \
    echo "Downloaded MeSH SKOS data ($(wc -l < /build/data/mesh-skos.ttl) lines)"

WORKDIR /build/mcp-rdf-explorer
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime - Minimal image with only necessary components
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /build/mcp-rdf-explorer/src/mcp-rdf-explorer /app
COPY --from=builder /build/data /app/data

# Ensure Python output is unbuffered (critical for MCP stdio communication)
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "server.py"]

# Default to using the built-in MeSH data
# Users can override with their own args
CMD ["--triple-file", "/app/data/mesh-skos.ttl"]
