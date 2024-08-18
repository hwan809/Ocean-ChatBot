import cv2
import numpy as np

background = cv2.imread('map.png')
background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

thickness = 10

def draw_line(img, pt1, pt2, color = (255, 38, 0)):
    if pt1[0] == pt2[0]:
        Pt1 = [pt1[0] - thickness, pt1[1]]
        Pt2 = [pt1[0] + thickness, pt1[1]]
        Pt3 = [pt2[0] + thickness, pt2[1]]
        Pt4 = [pt2[0] - thickness, pt2[1]]

        Pts = np.array([Pt1, Pt2, Pt3, Pt4])
        cv2.fillPoly(img, np.int32([Pts]), color)
    
    elif pt1[1] == pt2[1]:
        Pt1 = [pt1[0], pt1[1] - thickness]
        Pt2 = [pt1[0], pt1[1] + thickness]
        Pt3 = [pt2[0], pt2[1] + thickness]
        Pt4 = [pt2[0], pt2[1] - thickness]

        Pts = np.array([Pt1, Pt2, Pt3, Pt4])
        cv2.fillPoly(img, np.int32([Pts]), color)

def draw_triangle(img, point, d):
    if d == "left":
        pt = np.array([[point[0], point[1] - 30],
                       [point[0] - 40, point[1]],
                       [point[0], point[1] + 30]])
        cv2.fillPoly(img, [pt], (255, 38, 0))
    elif d == "right":
        pt = np.array([[point[0], point[1] - 30],
                       [point[0] + 40, point[1]],
                       [point[0], point[1] + 30]])
        cv2.fillPoly(img, [pt], (255, 38, 0))

# 블럭 하나 크기가 83

blockSize = 84

n = 10

pin = cv2.imread('pin.png')
pin = cv2.cvtColor(pin, cv2.COLOR_BGR2RGB)
ratio = 0.6
pin = cv2.resize(pin, dsize=(0, 0),fx = ratio, fy = ratio)

pinpin = np.round(np.array([309, 105]) * ratio).astype(np.uint8)

def makepin(img, pt):
    for i in range(pin.shape[0]):
        for j in range(pin.shape[1]):
            if (pin[i, j] != [0, 0, 0]).any():
                img[i - pinpin[0] + pt[1], j - pinpin[1] + pt[0]] = pin[i, j]
            else:
                continue

from copy import deepcopy

def arrow(line, n, d):
    edited = deepcopy(background)
    if line == 1:
        draw_line(edited, (1670, 1890), (1670, 1770))
        draw_line(edited, (1670 + thickness, 1770), (400, 1770))
        draw_line(edited, (400, 1770 + thickness), (400, 1770 - blockSize * n))

        if d == 'right':
            draw_line(edited, (400 - thickness, 1770 - blockSize * n), (450, 1770 - blockSize * n ))
            draw_triangle(edited, (450, 1770 - blockSize * n ), 'right')
            makepin(edited, (450 + 250, 1770 - blockSize * n - 20))

        elif d == 'left':
            draw_line(edited, (400 + thickness, 1770 - blockSize * n), (350, 1770 - blockSize * n ))
            draw_triangle(edited, (350, 1770 - blockSize * n ), 'left')
            makepin(edited, (350 - 250, 1770 - blockSize * n - 20))

    if line == 2:
        draw_line(edited, (1670, 1890), (1670, 1770))
        draw_line(edited, (1670 + thickness, 1770), (1035, 1770))
        draw_line(edited, (1035, 1770 + thickness), (1035, 1770 - blockSize * n))

        if d == 'right':
            draw_line(edited, (1035 - thickness, 1770 - blockSize * n), (1035 + 50, 1770 - blockSize * n))
            draw_triangle(edited, (1035 + 50, 1770 - blockSize * n ), 'right')
            makepin(edited, (1035 + 50 + 250, 1770 - blockSize * n - 20))

        elif d == 'left':
            draw_line(edited, (1035 + thickness, 1770 - blockSize * n), (1035 - 50, 1770 - blockSize * n))
            draw_triangle(edited, (1035 - 50, 1770 - blockSize * n ), 'left')
            makepin(edited, (1035 - 50 - 250, 1770 - blockSize * n - 20))

    if line == 3:
        draw_line(edited, (1670, 1890), (1670, 1770))
        draw_line(edited, (1670, 1770 + thickness), (1670, 1770 - blockSize * n))

        if d == 'right':
            draw_line(edited, (1670 - thickness, 1770 - blockSize * n), (1670 + 50, 1770 - blockSize * n))
            draw_triangle(edited, (1670 + 50, 1770 - blockSize * n ), 'right')
            makepin(edited, (1670 + 50 + 250, 1770 - blockSize * n - 20))

        elif d == 'left':
            draw_line(edited, (1670 + thickness, 1770 - blockSize * n), (1670 - 50, 1770 - blockSize * n))
            draw_triangle(edited, (1670 - 50, 1770 - blockSize * n ), 'left')
            makepin(edited, (1670 - 50 - 250, 1770 - blockSize * n - 20))
    
    return edited

booth = {}

for i in range(17):
    if i >= 0 and i <= 2:
        booth[f'C{13 + i}'] = (1, i, 'left')
    
    if i >= 3:
        booth['D' + f'{i - 2}'.zfill(2)] = (1, i, 'left')
    
    if i >= 0 and i <= 10:
        booth['A' + f'{i + 1}'.zfill(2)] = (3, i, 'right')
    
    if i >= 11:
        booth['B' + f'{i - 10}'.zfill(2)] = (3, i, 'right')


for i in range(15):
    if i >= 0:
        booth[f'B{39 + i}'] = (2, i + 1, 'left')
    
    if i >= 3:
        booth['C' + f'{i - 2}'.zfill(2)] = (1, i + 1, 'right')
    
    if i >= 0 and i <= 2:
        booth['B' + f'{i + 54}'.zfill(2)] = (1, i + 1, 'right')

for i in range(16):
    booth[f'B{23 + i}'] = (2, i + 1, 'right')
    booth['B' + f'{7 + i}'.zfill(2)] = (3, i + 1, 'left')


def get_location_image(team_code):
    team_cord = booth[team_code]

    img = arrow(team_cord[0], team_cord[1], team_cord[2])
    return img


