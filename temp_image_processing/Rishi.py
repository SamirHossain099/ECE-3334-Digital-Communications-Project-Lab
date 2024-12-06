import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform


def extract_worm_boundary(binary_img):
    # Make sure image is binary with values 0 and 255
    binary_img = (binary_img > 128).astype(np.uint8) * 255
    
    # Find contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Get the largest contour (should be the worm)
    worm_contour = max(contours, key=cv2.contourArea)
    
    # Create an empty image to draw the boundary
    boundary_img = np.zeros_like(binary_img)
    cv2.drawContours(boundary_img, [worm_contour], -1, 255, 1)
    
    # Create filled region image
    filled_img = np.zeros_like(binary_img)
    cv2.drawContours(filled_img, [worm_contour], -1, 255, -1)  # -1 for filling
    
    # Get boundary points as a list of [x,y] coordinates
    boundary_points = worm_contour.squeeze()
    
    return boundary_points, boundary_img, filled_img

def visualize_boundary_points(binary_img, boundary_points):
    """
    Visualize the boundary points to verify they're correct
    """
    vis_img = cv2.cvtColor(binary_img.copy(), cv2.COLOR_GRAY2BGR)
    
    # Draw each boundary point as a small red dot
    for point in boundary_points:
        cv2.circle(vis_img, tuple(point.astype(int)), 1, (0,0,255), -1)
    
    return vis_img

def test_boundary_extraction(image_path):
    # Read binary image
    binary_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if binary_img is None:
        raise ValueError(f"Failed to read image from path: {image_path}")
    
    # Extract boundary and filled region
    boundary_points, boundary_img, filled_img = extract_worm_boundary(binary_img)
    
    # Visualize boundary points
    points_vis = visualize_boundary_points(boundary_img, boundary_points)
    
    # Visualize all results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(221)
    plt.imshow(binary_img, cmap='gray')
    plt.title('Original Binary')
    
    plt.subplot(222)
    plt.imshow(boundary_img, cmap='gray')
    plt.title('Boundary')
    
    plt.subplot(223)
    plt.imshow(filled_img, cmap='gray')
    plt.title('Filled Region')
    
    plt.subplot(224)
    plt.imshow(cv2.cvtColor(points_vis, cv2.COLOR_BGR2RGB))
    plt.title('Boundary Points')
    
    plt.tight_layout()
    plt.show()
    
    return boundary_points, boundary_img, filled_img


def initialize_control_points(filled_img, num_points=500):

    y_coords, x_coords = np.where(filled_img > 0)
    points = np.column_stack((x_coords, y_coords))
    
    indices = np.random.choice(len(points), num_points, replace=False)
    sampled_points = points[indices]
    
    distances = squareform(pdist(sampled_points))
    mst = minimum_spanning_tree(distances).toarray()
    
    def bfs_get_farthest(graph, start):
        n = len(graph)
        dist = np.full(n, np.inf)
        parent = np.full(n, -1)
        dist[start] = 0
        
        queue = [start]
        while queue:
            current = queue.pop(0)
            for neighbor in range(n):
                if graph[current, neighbor] > 0 and dist[neighbor] == np.inf:
                    dist[neighbor] = dist[current] + 1
                    parent[neighbor] = current
                    queue.append(neighbor)
        
        end = np.argmax(dist)
        
        path = []
        current = end
        while current != -1:
            path.append(current)
            current = parent[current]
        
        return end, path[::-1], dist[end]  
    
    end1, _, _ = bfs_get_farthest(mst, 0)   # Added mst here
    
    end2, path, max_dist = bfs_get_farthest(mst, end1) 
    
    print(f"End points of diameter: end1 = {end1}, end2 = {end2}")
    print(f"Path length: {len(path)}")
    print(f"Path: {path}")
    print(f"Maximum distance: {max_dist}")
    
    debug_img = cv2.cvtColor(filled_img.copy(), cv2.COLOR_GRAY2BGR)
    
    for point in sampled_points:
        cv2.circle(debug_img, tuple(point.astype(int)), 2, (255,0,0), -1)
    
    for i in range(len(sampled_points)):
        for j in range(i+1, len(sampled_points)):
            if mst[i,j] != 0:
                pt1 = tuple(sampled_points[i].astype(int))
                pt2 = tuple(sampled_points[j].astype(int))
                cv2.line(debug_img, pt1, pt2, (0,255,0), 1)
    
    cv2.circle(debug_img, tuple(sampled_points[end1].astype(int)), 5, (255,0,255), -1)  
    cv2.circle(debug_img, tuple(sampled_points[end2].astype(int)), 5, (255,255,0), -1)  
    
    if len(path) > 1:
        for i in range(len(path)-1):
            pt1 = tuple(sampled_points[path[i]].astype(int))
            pt2 = tuple(sampled_points[path[i+1]].astype(int))
            cv2.circle(debug_img, pt1, 4, (0,0,255), -1)
            cv2.line(debug_img, pt1, pt2, (0,0,255), 2)
        
        pt = tuple(sampled_points[path[-1]].astype(int))
        cv2.circle(debug_img, pt, 4, (0,0,255), -1)
    
    plt.figure(figsize=(8,8))
    plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
    plt.title('MST and Diameter Path')
    plt.show()
    
    return sampled_points[path]




# Run the test
boundary_points, boundary_img, filled_img = test_boundary_extraction("temp_image_processing/frame10.png")

# Test the initialization
sampled_points = initialize_control_points(filled_img)