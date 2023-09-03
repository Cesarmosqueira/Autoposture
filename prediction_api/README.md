## Prediction System API

Build and run the docker image
(change the port inside the dockerfile if needed)
```
docker build -t ap_predictions .
docker run -d -p 8080:8080 ap_predictions
```

to check if server is running correctly send a GET request to /healthcheck
```
curl http://localhost:8080/healthcheck
```

Now the `/predict` endpoint is available and you can run the main script \
[back](https://github.com/Cesarmosqueira/Autoposture/tree/master#readme)
