mkdir -p nginx_conf
conf_file=nginx_conf/nginx.conf
SERVER_NUM=$1

touch ${conf_file}
cat > ${conf_file} <<EOF
worker_processes auto;
events {
    worker_connections 1024;
}

http {
    upstream backend {
        least_conn;
EOF

for (( idx = 0; idx < SERVER_NUM; ++idx )); do
    cat >> ${conf_file} <<EOF
        server vllm_$idx:8000 max_fails=3 fail_timeout=10s;
EOF
done

cat >> ${conf_file} <<EOF
    }

    server {
        listen 80;
        location / {
            proxy_pass http://backend;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }
    }
}
EOF
