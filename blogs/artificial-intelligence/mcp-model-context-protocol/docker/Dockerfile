FROM rocm/vllm:latest

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /

# Install node.js see:
# https://github.com/nodesource/distributions?tab=readme-ov-file#installation-instructions-deb

RUN apt-get update \
    && curl -fsSL https://deb.nodesource.com/setup_23.x -o nodesource_setup.sh \
    && sudo -E bash nodesource_setup.sh \
    && sudo apt-get install -y nodejs

# Install MCP and MCP servers
RUN pip install "mcp[cli]"

RUN pip install mcp-server-sqlite

RUN git clone https://github.com/TimLukaHorstmann/mcp-weather.git \
    && cd mcp-weather \
    && npm install \
    && npm run build    

RUN git clone https://github.com/openbnb-org/mcp-server-airbnb.git \
    && cd mcp-server-airbnb \
    && npm install \
    && npm run build

CMD ["/bin/bash"]