import cv2
from inference_sdk import InferenceHTTPClient

# Load image
image_path = "WIN_20250621_09_49_59_Pro.jpg"
img = cv2.imread(image_path)

# Initialize client
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="9k6SBo0cQlOkFConFfVR"
)

# Run workflow
result = client.run_workflow(
    workspace_name="anthony-intj3",
    workflow_id="custom-workflow-3",
    images={"image": image_path},
    use_cache=True
)

# Plot points
predictions = result[0]['predictions']['predictions']
threshold = 0.4
for pred in predictions:
    # Adjust according to what the response contains
    print(pred)
    if pred['confidence']<threshold:
        continue
    x, y = int(pred['x']), int(pred['y'])  # center of bbox
    w, h = int(pred['width']), int(pred['height'])
    label = pred.get("class", "")

    # Calculate top-left and bottom-right corners of bounding box
    x1 = x - w // 2
    y1 = y - h // 2
    x2 = x + w // 2
    y2 = y + h // 2

    # Draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw label
    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 0, 0), 1)

img = cv2.resize(img,(1280,720))
# Display
cv2.imshow("Detections", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
