import os

import cv2

link_pairs = [[0, 1], [1, 2], [2, 6], 
              [3, 6], [3, 4], [4, 5], 
              [6, 7], [7,12], [11, 12], 
              [10, 11], [7, 13], [13, 14],
              [14, 15],[7, 8],[8, 9]]

link_color = [(0, 0, 255), (0, 0, 255), (0, 0, 255),
              (0, 255, 0), (0, 255, 0), (0, 255, 0),
              (0, 255, 255), (0, 0, 255), (0, 0, 255),
              (0, 0, 255), (0, 255, 0), (0, 255, 0),
              (0, 255, 0), (0, 255, 255), (0, 255, 255)]

point_color = [(255,0,0),(0,255,0),(0,0,255), 
               (128,0,0), (0,128,0), (0,0,128),
               (255, 255, 0),(0,255,255),(255, 0, 255),
               (128,128,0),(0, 128, 128),(128,0,128),
               (128,255,0),(128,128,128),(255,128,0),
               (255,0,128),(255,255,255)]


coco_link_pairs = [[0, 1], [1, 2], [2, 3], 
              [3, 4], [1, 5], [5, 6], 
              [6, 7], [1, 8], [8, 9], 
              [9, 10], [1, 11], [11, 12],
              [12, 13],[0, 14],[14, 16], [0, 15], [15, 17]]

coco_link_color = [(0, 0, 255), (0, 0, 255), (0, 0, 255),
              (0, 255, 0), (0, 255, 0), (0, 255, 0),
              (0, 255, 255), (0, 0, 255), (0, 0, 255),
              (0, 0, 255), (0, 255, 0), (0, 255, 0),
              (0, 255, 0), (0, 255, 255), (0, 255, 255), (128, 128, 0), (128, 0, 128)]

coco_point_color = [(255,0,0),(0,255,0),(0,0,255), 
               (128,0,0), (0,128,0), (0,0,128),
               (255, 255, 0),(0,255,255),(255, 0, 255),
               (128,128,0),(0, 128, 128),(128,0,128),
               (128,255,0),(128,128,128),(255,128,0),
               (255,0,128),(255,255,255), (128, 128, 0), (128, 0, 128)]

'''
original pose:
2 1 0 : right leg
3 4 5 : left leg
12 11 10 : right arm
13 14 15 : left arm
7 : neck
9 8 : head

coco:
0 : nose
1 : neck
2 3 4 : right arm
5 6 7 : left arm
8 9 10 : right leg
11 12 13 : left leg
14 15 : eyes
16 17 : ears
'''

def convert_pose(pose):
    def get_length(p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

    coco = []
    
    nose = ((pose[9][0] + pose[8][0]) // 2, (pose[9][1] + pose[8][1]) // 2)
    eye_center = ((pose[9][0] * 6 + pose[8][0] * 4) // 10, (pose[9][1] * 6 + pose[8][1] * 4) // 10)
    coco.append(nose)
    coco.append(pose[7])
    coco += reversed(pose[10:13])
    coco += pose[13:16]
    coco += reversed(pose[0:3])
    coco += pose[3:6]

    ear_dist = ((coco[2][0] - coco[5][0]) // 4, (coco[2][1] - coco[5][1]) // 4)

    xdelta = (coco[0][0] - coco[1][0]) - (coco[1][0] - (coco[11][0] + coco[8][0]) // 2) // 10

    collar_vec = (coco[2][0] - coco[1][0], coco[2][1] - coco[1][1])
    neck_vec = (coco[0][0] - coco[1][0], coco[0][1] - coco[1][1])

    ydelta = collar_vec[0] * neck_vec[0] + collar_vec[1] * neck_vec[1]

    coco.append(((eye_center[0] + ear_dist[0] // 3 - xdelta // 3), (eye_center[1] + ear_dist[1] // 3 - ydelta // 100)))
    coco.append(((eye_center[0] - ear_dist[0] // 3 - xdelta // 3), (eye_center[1] - ear_dist[1] // 3 - ydelta // 100)))
    coco.append(((coco[0][0] + ear_dist[0] - xdelta // 2), (coco[0][1] + ear_dist[1] - ydelta // 150)))
    coco.append(((coco[0][0] - ear_dist[0] - xdelta // 2), (coco[0][1] - ear_dist[1] - ydelta // 150)))

    return coco


def vis_pose(image_path, pose_keypoints, show=True, resized=False):
    image = cv2.imread(image_path)

    if resized:
        pose_keypoints, _, _, _, _ = resize_pose(pose_keypoints[:-1])

    for idx, pair in enumerate(link_pairs):
        cv2.line(image, pose_keypoints[pair[0]], pose_keypoints[pair[1]], link_color[idx], 2)

    for idx, point in enumerate(pose_keypoints):
        cv2.putText(image,str(idx),point,cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        if idx != 16:
            cv2.circle(image, point, 5, point_color[idx], thickness=-1)
        else:
            cv2.circle(image, point, 20, point_color[idx], thickness=-1)

    if show:
        cv2.imshow("image", image)
        cv2.moveWindow("image", 0, 0)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image

def vis_pose_coco(image_path, pose_keypoints, show=True, resized=False):
    image = cv2.imread(image_path)

    pose_keypoints = convert_pose(pose_keypoints)

    if resized:
        pose_keypoints, _, _, _, _ = resize_pose(pose_keypoints)

    for idx, pair in enumerate(coco_link_pairs):
        cv2.line(image, pose_keypoints[pair[0]], pose_keypoints[pair[1]], coco_link_color[idx], 2)

    for idx, point in enumerate(pose_keypoints):
        cv2.putText(image,str(idx),point,cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        cv2.circle(image, point, 5, coco_point_color[idx], thickness=-1)

    if show:
        cv2.imshow("image", image)
        cv2.moveWindow("image", 0, 0)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image

def resize_pose(pose_keypoints):
    width = 200
    padding = 10
    min_x = 10000
    min_y = 10000
    max_x = 0
    max_y = 0

    for point in pose_keypoints:
        min_x = min(min_x, point[0])
        min_y = min(min_y, point[1])
        max_x = max(max_x, point[0])
        max_y = max(max_y, point[1])
    
    pose_width = max_x - min_x
    pose_height = max_y - min_y

    new_keypoints = []

    for point in pose_keypoints:
        if pose_width > pose_height:
            new_keypoints.append(
                (int((width/2 + (point[0] - (max_x + min_x)/2) / pose_width * (width - 2*padding))*0.9),
                int((20 + width/2 + (point[1] - (max_y + min_y)/2) / pose_width * (width - 2*padding))*1.05))
            )
        else:
            new_keypoints.append(
                (int((width/2 + (point[0] - (max_x + min_x)/2) / pose_height * (width - 2*padding))*0.9),
                int((20 + width/2 + (point[1] - (max_y + min_y)/2) / pose_height * (width - 2*padding))*1.05))
            )

    keypoints_y = [i[1] for i in new_keypoints]
    keypoints_x = [i[0] for i in new_keypoints]

    
    print('keypoints_y = "', keypoints_y, '"')
    print('keypoints_x = "', keypoints_x, '"\n')

    return new_keypoints, min_x, min_y, pose_width, pose_height


def vis_pose_data(data):
    image_dir_path = './affordance_data/data'
    data = data.split(' ')
    image_path = os.path.join(image_dir_path, data[0])
    image_cluster = eval(data[-1])
    pose_data = data[1:-1]
    pose_data = [int(eval(x)) for x in pose_data]
    pose_keypoints = []
    for i in range(0, len(pose_data), 2):
        pose_keypoints.append((pose_data[i], pose_data[i+1]))

    vis_pose(image_path, pose_keypoints)
    vis_pose(image_path, pose_keypoints, resized=True)

    vis_pose_coco(image_path, pose_keypoints)
    vis_pose_coco(image_path, pose_keypoints, resized=True)

if __name__ == '__main__':
    predicted = [(595.5921267553274, 404.10781700000007), (587.4540269712579, 375.26168899999993), (674.1323874006324, 382.447135), (709.0841106363025, 390.41310699999985), (738.5057531861758, 386.341047), (664.3825897951702, 417.24250400000005), (693.034285958547, 387.2644009999999), (714.1887366306264, 310.66663199999994), (711.364003176137, 296.50660700000003), (702.7690875859935, 228.0), (630.6202741684986, 387.9697880000001), (647.1081440971391, 359.6237799999999), (670.0994861445556, 309.9926839999998), (761.1125559742334, 314.1373520000001), (758.0154726576313, 389.9528070000001), (678.8184304028331, 402.0585960000001)]
    gt = [(613, 410), (603, 374), (646, 367), (671, 370), (673, 389), (630, 415), (660, 370), (672, 298), (670, 276), (665, 228), (638, 382), (631, 340), (642, 298), (703, 298), (693, 354), (640, 379)]
    predicted = [(round(x[0]), round(x[1])) for x in predicted]

    # vis_pose(os.path.join('./affordance_data/data', 'ELR_matches/S02/E0012.mkv/frame_00002381.jpg'), predicted + [(0, 0)])
    # vis_pose(os.path.join('./affordance_data/data', 'ELR_matches/S02/E0012.mkv/frame_00002381.jpg'), gt + [(0, 0)])
    # vis_pose_coco(os.path.join('./affordance_data/data', 'ELR_matches/S02/E0012.mkv/frame_00002381.jpg'), gt + [(0, 0)])

    train_data_path = './affordance_data/trainlist.txt'
    train_data = []
    with open(train_data_path, 'r') as f:
        train_data = list(f.readlines())

    for data in train_data:
        vis_pose_data(data)