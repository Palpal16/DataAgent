FROM debian:bookworm-slim AS build

RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY src ./src

# Try static first for portability; fall back to dynamic if static fails
RUN g++ -O2 -std=c++17 -static -s -o /usr/local/bin/resultcmp ./src/main.cpp || \
    g++ -O2 -std=c++17 -o /usr/local/bin/resultcmp ./src/main.cpp

FROM debian:bookworm-slim
COPY --from=build /usr/local/bin/resultcmp /usr/local/bin/resultcmp

# Working directory where you'll mount your host project
WORKDIR /work
ENTRYPOINT ["/usr/local/bin/resultcmp"]


