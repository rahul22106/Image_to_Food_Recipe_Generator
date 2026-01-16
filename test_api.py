import requests
import json
from pathlib import Path


API_URL = "http://localhost:8000"


def test_health():
    print("\n" + "="*70)
    print("Testing Health Check Endpoint")
    print("="*70)
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200


def test_info():
    print("\n" + "="*70)
    print("Testing Info Endpoint")
    print("="*70)
    
    response = requests.get(f"{API_URL}/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200


def test_predict(image_path: str, top_k: int = 5):
    print("\n" + "="*70)
    print("Testing Predict Endpoint")
    print("="*70)
    print(f"Image: {image_path}")
    print(f"Top-K: {top_k}")
    
    if not Path(image_path).exists():
        print(f"❌ Error: Image not found: {image_path}")
        return False
    
    with open(image_path, 'rb') as f:
        files = {'file': (Path(image_path).name, f, 'image/jpeg')}
        params = {'top_k': top_k}
        
        response = requests.post(
            f"{API_URL}/predict",
            files=files,
            params=params
        )
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Success: {data['success']}")
        print(f"Message: {data['message']}")
        print(f"Total Predictions: {data['total_predictions']}")
        print("\nTop Predictions:")
        
        for pred in data['predictions'][:3]:
            print(f"\n  {pred['rank']}. {pred['name']}")
            print(f"     Similarity: {pred['similarity_score']:.4f}")
            print(f"     Ingredients: {pred['ingredients'][:100]}...")
        
        return True
    else:
        print(f"❌ Error: {response.text}")
        return False


def test_batch_predict(image_paths: list, top_k: int = 3):
    print("\n" + "="*70)
    print("Testing Batch Predict Endpoint")
    print("="*70)
    print(f"Images: {len(image_paths)}")
    print(f"Top-K: {top_k}")
    
    files = []
    for img_path in image_paths:
        if Path(img_path).exists():
            files.append(
                ('files', (Path(img_path).name, open(img_path, 'rb'), 'image/jpeg'))
            )
    
    if not files:
        print("❌ Error: No valid images found")
        return False
    
    params = {'top_k': top_k}
    
    response = requests.post(
        f"{API_URL}/predict/batch",
        files=files,
        params=params
    )
    
    for _, (_, f, _) in files:
        f.close()
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Success: {data['success']}")
        print(f"Message: {data['message']}")
        print(f"\nResults:")
        
        for result in data['results']:
            print(f"\n  📷 {result['filename']}")
            if result['success']:
                top_pred = result['predictions'][0]
                print(f"     Top Recipe: {top_pred['name']}")
                print(f"     Score: {top_pred['similarity_score']:.4f}")
            else:
                print(f"     ❌ Error: {result.get('error', 'Unknown error')}")
        
        return True
    else:
        print(f"❌ Error: {response.text}")
        return False


def run_all_tests():
    print("\n" + "="*70)
    print("RECIPE GENERATOR API - TEST SUITE")
    print("="*70)
    
    print("\n⚠️  Make sure the API server is running:")
    print("   python app.py")
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    input()
    
    results = {}
    
    results['health'] = test_health()
    results['info'] = test_info()
    
    test_images = list(Path("artifacts/data/processed/images").glob("*.jpg"))[:3]
    
    if test_images:
        results['predict'] = test_predict(str(test_images[0]), top_k=5)
        
        if len(test_images) >= 2:
            results['batch'] = test_batch_predict([str(img) for img in test_images[:2]], top_k=3)
    else:
        print("\n⚠️  No test images found in artifacts/data/processed/images/")
        results['predict'] = False
        results['batch'] = False
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name.upper()}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*70 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "health":
            test_health()
        elif command == "info":
            test_info()
        elif command == "predict" and len(sys.argv) > 2:
            test_predict(sys.argv[2], top_k=int(sys.argv[3]) if len(sys.argv) > 3 else 5)
        elif command == "batch" and len(sys.argv) > 2:
            test_batch_predict(sys.argv[2:], top_k=3)
        else:
            print("Usage:")
            print("  python test_api.py health")
            print("  python test_api.py info")
            print("  python test_api.py predict <image_path> [top_k]")
            print("  python test_api.py batch <image1> <image2> ...")
    else:
        run_all_tests()