"""
REST resource for capturing simulation visualization as images.
Provides endpoint for capturing current simulation state in PNG, JPEG, or GIF format.
"""

from flask import request, send_file
from flask_restful import Resource

from api.simulation_instance import sim_api
from api.util.image_capture import ImageCaptureManager


class ImageCaptureResource(Resource):
    """
    REST resource for capturing simulation images.

    Endpoint: GET /api/capture

    Query Parameters:
        format (str): Image format - 'png', 'jpeg', or 'gif' (default: 'png')
        width (int): Image width in pixels (default: 1920, range: 320-3840)
        height (int): Image height in pixels (default: 1080, range: 240-2160)
        view_type (str): View type - '2d', '3d', 'side', or 'multi' (default: '3d')
        quality (int): JPEG quality 1-100 (default: 85, only for JPEG format)

    Returns:
        Binary image data with appropriate Content-Type header

    Error Responses:
        404: No simulation loaded
        400: Invalid parameters
        500: Internal server error during rendering
    """

    def __init__(self):
        """Initialize the image capture resource with a capture manager."""
        super().__init__()
        self.capture_manager = ImageCaptureManager(max_frames=10)

    def get(self):
        """
        Capture current simulation state as image.

        Returns:
            Flask response with binary image data or JSON error
        """
        # Check if simulation exists
        if sim_api.simulation is None:
            return {'error': 'No simulation loaded'}, 404

        # Parse and validate format parameter
        format = request.args.get('format', 'png').lower()
        if format not in ['png', 'jpeg', 'gif']:
            return {
                'error': 'Invalid format. Must be png, jpeg, or gif',
                'received': format
            }, 400

        # Parse and validate view_type parameter
        view_type = request.args.get('view_type', '3d').lower()
        if view_type not in ['2d', '3d', 'side', 'multi']:
            return {
                'error': 'Invalid view_type. Must be 2d, 3d, side, or multi',
                'received': view_type
            }, 400

        # Parse and validate numeric parameters
        try:
            width = int(request.args.get('width', 1920))
            height = int(request.args.get('height', 1080))
            quality = int(request.args.get('quality', 85))

            # Validate ranges
            if not (320 <= width <= 3840):
                return {
                    'error': 'Width must be between 320 and 3840',
                    'received': width
                }, 400

            if not (240 <= height <= 2160):
                return {
                    'error': 'Height must be between 240 and 2160',
                    'received': height
                }, 400

            if not (1 <= quality <= 100):
                return {
                    'error': 'Quality must be between 1 and 100',
                    'received': quality
                }, 400

        except ValueError as e:
            return {
                'error': 'Invalid numeric parameter',
                'details': str(e)
            }, 400

        # Capture frame
        try:
            image_data = self.capture_manager.capture_frame(
                simulation=sim_api.simulation,
                format=format,
                width=width,
                height=height,
                view_type=view_type,
                quality=quality
            )

            # Determine MIME type
            mime_types = {
                'png': 'image/png',
                'jpeg': 'image/jpeg',
                'gif': 'image/gif'
            }

            return send_file(
                image_data,
                mimetype=mime_types[format],
                as_attachment=False,
                download_name=f'simulation.{format}'
            )

        except ValueError as e:
            # Handle specific value errors (e.g., empty buffer for GIF)
            return {
                'error': str(e)
            }, 400

        except Exception as e:
            # Handle any other rendering errors
            return {
                'error': f'Failed to capture image: {str(e)}',
                'type': type(e).__name__
            }, 500
