from inference_sdk import InferenceHTTPClient
import cv2

image_path = "WIN_20250620_23_36_09_Pro.jpg"
img = cv2.imread(image_path)

client = InferenceHTTPClient(
    api_url="http://localhost:9001", # use local inference server
    api_key="9k6SBo0cQlOkFConFfVR"
)

result = client.run_workflow(
    workspace_name="anthony-intj3",
    workflow_id="custom-workflow-3",
    images={
        "image": image_path
    }, 
    use_cache=True
)

predictions = result[0]['predictions']['predictions']
threshold = 0.5
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