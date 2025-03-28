FROM nginx:alpine

WORKDIR /usr/share/nginx/html

COPY . .

EXPOSE 3000

# แก้ไข port เป็น 3000
RUN sed -i 's/listen       80/listen       3000/g' /etc/nginx/conf.d/default.conf
RUN sed -i 's/listen  \[::\]:80/listen  \[::\]:3000/g' /etc/nginx/conf.d/default.conf

CMD ["nginx", "-g", "daemon off;"]
