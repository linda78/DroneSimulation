"""
Custom drone visualization with textured mesh
"""
import numpy as np
from PIL import Image
from vispy import gloo
from vispy.scene import visuals, transforms
from typing import Optional
import os

from gui import lighten_color


def load_drone_texture(image_path: str, size: int = 256) -> Optional[gloo.Texture2D]:
    """
    Load drone image (GIF/PNG) as OpenGL texture

    Args:
        image_path: Path to image file
        size: Target size for texture (square)

    Returns:
        Texture2D object or None if loading fails
    """
    if not image_path or not os.path.exists(image_path):
        return None

    try:
        # Load image with PIL
        img = Image.open(image_path)

        # For GIF, get first frame
        if hasattr(img, 'seek'):
            img.seek(0)

        # Convert to RGBA
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        # Make everything fully opaque (no transparency)
        img_array = np.array(img)
        img_array[:, :, 3] = 255  # Set alpha to 255 (fully opaque) for all pixels
        img = Image.fromarray(img_array)

        # Resize to square
        img = img.resize((size, size), Image.Resampling.LANCZOS)

        # Convert to numpy array (uint8)
        img_array = np.array(img, dtype=np.uint8)

        # Create OpenGL texture
        texture = gloo.Texture2D(img_array, interpolation='linear')

        return texture

    except Exception as e:
        print(f"Warning: Could not load drone texture from {image_path}: {e}")
        return None


def create_textured_quad(size: float = 1.0):
    """
    Create quad mesh geometry with texture coordinates

    Args:
        size: Size of the quad

    Returns:
        Tuple of (vertices, faces, texcoords)
    """
    half = size / 2.0

    # Quad vertices (centered at origin, in XY plane)
    vertices = np.array([
        [-half, -half, 0],  # Bottom-left
        [half, -half, 0],   # Bottom-right
        [half, half, 0],    # Top-right
        [-half, half, 0]    # Top-left
    ], dtype=np.float32)

    # Two triangles to form quad
    faces = np.array([
        [0, 1, 2],  # First triangle
        [0, 2, 3]   # Second triangle
    ], dtype=np.uint32)

    # Texture coordinates (standard UV mapping)
    texcoords = np.array([
        [0, 0],  # Bottom-left
        [1, 0],  # Bottom-right
        [1, 1],  # Top-right
        [0, 1]   # Top-left
    ], dtype=np.float32)

    return vertices, faces, texcoords


class TexturedMeshVisual(visuals.Mesh):
    """
    Mesh visual with texture support and color tinting
    """

    def __init__(self, texture: gloo.Texture2D, size: float = 1.0,
                 color: tuple = (1.0, 1.0, 1.0), **kwargs):
        """
        Initialize textured mesh

        Args:
            texture: OpenGL texture
            size: Size of mesh
            color: RGB color for tinting (0-1 range)
            **kwargs: Additional arguments for Mesh
        """
        vertices, faces, texcoords = create_textured_quad(size)

        # Initialize mesh
        super().__init__(
            vertices=vertices,
            faces=faces,
            **kwargs
        )

        # Unfreeze to allow adding custom attributes
        self.unfreeze()

        # Store texture and geometry data
        self._texture = texture
        self._vertices_data = vertices
        self._texcoords = texcoords
        self._tint_color = np.array(color[:3], dtype=np.float32)

        # Apply texture via custom shader
        self._setup_texture_shader()

        # Refreeze
        self.freeze()

    def _setup_texture_shader(self):
        """Setup custom shader for textured rendering"""

        # Vertex shader with VisPy's transform template
        vertex_shader = """
        attribute vec3 a_position;
        attribute vec2 a_texcoord;

        varying vec2 v_texcoord;

        void main() {
            vec4 pos = vec4(a_position, 1.0);
            gl_Position = $transform(pos);
            v_texcoord = a_texcoord;
        }
        """

        # Fragment shader
        fragment_shader = """
        uniform sampler2D u_texture;

        varying vec2 v_texcoord;

        void main() {
            vec4 tex_color = texture2D(u_texture, v_texcoord);
            gl_FragColor = tex_color;
        }
        """

        # Create shader program
        self.shared_program.vert = vertex_shader
        self.shared_program.frag = fragment_shader

        # Set texture coordinates as attribute
        self.shared_program['a_texcoord'] = gloo.VertexBuffer(self._texcoords)

        # Set texture uniform
        self.shared_program['u_texture'] = self._texture

        # Enable alpha blending for transparency
        self.set_gl_state('translucent', depth_test=True, cull_face=False)

    def _update_data(self):
        """Override to set vertex data for our custom shader"""
        # Set vertices attribute for our shader
        self.shared_program['a_position'] = gloo.VertexBuffer(self._vertices_data)
        return True


class OBJMeshVisual(visuals.Mesh):
    """
    3D mesh visual loaded from OBJ file with texture support using TextureFilter
    """

    def __init__(self, mesh_path: str, texture_path: Optional[str] = None,
                 scale: float = 1.0, color: tuple = (1.0, 1.0, 1.0), **kwargs):
        """
        Initialize OBJ mesh visual

        Args:
            mesh_path: Path to .obj or .obj.gz file
            texture_path: Path to texture image (optional)
            scale: Scaling factor for the mesh
            color: RGB color for tinting (0-1 range)
            **kwargs: Additional arguments for Mesh
        """
        from vispy.io import imread, read_mesh
        from vispy.visuals.filters import TextureFilter

        # Load mesh using VisPy's read_mesh (gets normals and texcoords)
        vertices, faces, normals, texcoords = read_mesh(mesh_path)

        # Center mesh at origin BEFORE scaling
        center = vertices.mean(axis=0)
        vertices = vertices - center

        # Then scale
        vertices = vertices * scale

        print(f"  Mesh bounds after centering and scaling: min={vertices.min(axis=0)}, max={vertices.max(axis=0)}")

        # Initialize base mesh with smooth shading for better lighting
        super().__init__(vertices=vertices, faces=faces, shading='smooth', color=lighten_color(color, 0.6), **kwargs)

        # self.transform = transforms.MatrixTransform()
        # Set shininess for better appearance
        if hasattr(self, 'shading_filter') and self.shading_filter is not None:
            self.shading_filter.shininess = 1e+1

        # Load and apply texture if provided
        if texture_path and os.path.exists(texture_path):
            try:
                texture = np.flipud(imread(texture_path))
                if texcoords is not None:
                    print(f"  Texture loaded successfully from {texture_path}")
                    # Apply texture using TextureFilter
                    texture_filter = TextureFilter(texture, texcoords)
                    self.attach(texture_filter)
                else:
                    print(f"  Warning: No texture coordinates in mesh")
            except Exception as e:
                print(f"  Warning: Could not load texture from {texture_path}: {e}")
        else:
            if texture_path:
                print(f"  Warning: Texture file not found: {texture_path}")

