from locust import HttpUser, task
import json, random

cities = [
        "Tokyo, Japan",
        "Delhi, India",
        "Shanghai, China",
        "São Paulo, Brazil",
        "Mexico City, Mexico",
        "New York City, USA",
        "London, UK",
        "Beijing, China",
        "Mumbai, India",
        "Osaka, Japan",
        "Cairo, Egypt",
        "Dhaka, Bangladesh",
        "Istanbul, Turkey",
        "Buenos Aires, Argentina",
        "Paris, France",
        "Seoul, South Korea",
        "Karachi, Pakistan",
        "Lagos, Nigeria",
        "Bangkok, Thailand",
        "Manila, Philippines"
        ]

class MyRayServeUser(HttpUser):
    @task
    def post_with_headers(self):
        
        from_city = cities[random.randint(0,19)]
        to_city = cities[random.randint(0,19)]

        REQUEST_PAYLOAD = { "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "messages": [{"role": "user", "content": f"How do you travel from {from_city} to {to_city} by train?"}]
        }
        REQUEST_HEADERS = {
            "Content-Type": "application/json",
            "Authorization": "Bearer fake-key"
        }

        # Convert payload to JSON string if Content-Type is application/json
        import json
        payload = json.dumps(REQUEST_PAYLOAD)

        self.client.post(
            "/v1/chat/completions",
            data=payload,
            headers=REQUEST_HEADERS
        )