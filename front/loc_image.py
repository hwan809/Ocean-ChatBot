from PIL import Image

TEAMCODE_DB = {
    'A01': (200, 200)
}

def get_location_image(team_code):
    location = (125, 720)

    if team_code in TEAMCODE_DB:
        location = TEAMCODE_DB[team_code]

    base_img = Image.open('data/ocean.png')
    pin_img = Image.open('data/placeholder.png').convert('RGBA')
    pin_img_w, pin_img_h = pin_img.size
    
    location = (location[0] - pin_img_w // 2, location[1] - pin_img_h // 2)
    base_img.paste(pin_img, location, pin_img)

    return base_img
