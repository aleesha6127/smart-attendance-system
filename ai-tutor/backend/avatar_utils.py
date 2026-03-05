from PIL import Image, ImageDraw, ImageFont
import os
import hashlib
import io

def generate_avatar(name, size=(100, 100)):
    """
    Generate a profile avatar based on the user's name initials
    """
    # Create a colored square
    color_map = {
        'A': (255, 99, 132),   # Red
        'B': (54, 162, 235),   # Blue
        'C': (255, 205, 86),   # Yellow
        'D': (75, 192, 192),   # Teal
        'E': (153, 102, 255),  # Purple
        'F': (255, 159, 64),   # Orange
        'G': (199, 199, 199),  # Gray
        'H': (83, 102, 147),   # Dark Blue
        'I': (255, 99, 71),    # Tomato
        'J': (60, 179, 113),   # Medium Sea Green
        'K': (218, 165, 32),   # Goldenrod
        'L': (138, 43, 226),   # Blue Violet
        'M': (220, 20, 60),    # Crimson
        'N': (30, 144, 255),   # Dodger Blue
        'O': (255, 165, 0),    # Orange
        'P': (128, 0, 128),    # Purple
        'Q': (255, 192, 203),  # Pink
        'R': (106, 90, 205),   # Slate Blue
        'S': (255, 140, 0),    # Dark Orange
        'T': (0, 128, 0),      # Green
        'U': (184, 134, 11),   # Dark Goldenrod
        'V': (221, 160, 221),  # Plum
        'W': (139, 69, 19),    # Saddle Brown
        'X': (47, 79, 79),     # Dark Slate Gray
        'Y': (255, 215, 0),    # Gold
        'Z': (0, 100, 0)       # Dark Green
    }
    
    # Get the first letter of the name to determine color
    first_letter = name[0].upper() if name else 'U'
    color = color_map.get(first_letter, (128, 128, 128))  # Default gray
    
    # Create image
    image = Image.new('RGB', size, color=color)
    draw = ImageDraw.Draw(image)
    
    # Get initials (first letter of first name and first letter of last name)
    parts = name.split()
    if len(parts) >= 2:
        initials = parts[0][0].upper() + parts[-1][0].upper()
    else:
        initials = parts[0][0].upper() if parts else 'U'
    
    # Try to use a system font, fallback to default if not available
    try:
        # Try common font paths on Windows
        font = ImageFont.truetype("arial.ttf", size[0] // 2)
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size[0] // 2)
        except:
            # Use default font if Arial is not available
            font = ImageFont.load_default()
    
    # Calculate text position to center it
    bbox = draw.textbbox((0, 0), initials, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    
    # Draw the text
    draw.text((x, y), initials, fill=(255, 255, 255), font=font)
    
    # Save the image
    avatar_filename = f"{hashlib.md5(name.encode()).hexdigest()}_{size[0]}x{size[1]}.png"
    avatar_path = os.path.join('static', 'avatars', avatar_filename)
    
    # Ensure the avatars directory exists
    os.makedirs(os.path.dirname(avatar_path), exist_ok=True)
    
    image.save(avatar_path)
    
    return f"/{avatar_path}"


def get_default_avatar_url(name):
    """
    Generate an avatar if it doesn't exist and return the URL
    """
    # Create a hash-based filename to ensure consistent naming
    avatar_filename = f"{hashlib.md5(name.encode()).hexdigest()}_100x100.png"
    avatar_path = os.path.join('static', 'avatars', avatar_filename)
    
    # If avatar doesn't exist, generate it
    if not os.path.exists(avatar_path):
        return generate_avatar(name)
    
    return f"/{avatar_path}"