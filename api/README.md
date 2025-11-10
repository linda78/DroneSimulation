# REST API Documentation

This directory contains the REST API server for controlling the Drone Simulation.

## Files

### `server.py`
Main Flask application and REST API endpoints.

**Key Components:**
- `SimulationAPI` - Core simulation control class
- `SimulationResource` - Handles simulation CRUD operations
- `SimulationControlResource` - Handles simulation playback control (start/stop/reset/step)
- `DronesResource` - Provides drone information
- `HistoryResource` - Provides simulation history
- `StatusResource` - Provides simulation status
- `create_app()` - Flask app factory
- `run_server()` - Server runner

### `swagger_spec.py`
Comprehensive Swagger/OpenAPI specification with detailed examples.

**Contains:**
- `SWAGGER_CONFIG` - Swagger UI configuration
- `SWAGGER_TEMPLATE` - OpenAPI template with API metadata
- `ENDPOINT_SPECS` - Detailed endpoint specifications with examples

**Features:**
- Complete request/response examples for every endpoint
- Multiple example scenarios (success, errors, edge cases)
- Detailed parameter descriptions
- Use case documentation
- Error handling examples

## API Endpoints

### Status
- `GET /api/status` - Get simulation status

### Simulation Management
- `GET /api/simulation` - Get current simulation state
- `POST /api/simulation` - Create/load simulation
- `DELETE /api/simulation` - Stop and clear simulation

### Simulation Control
- `POST /api/simulation/control/start` - Start simulation
- `POST /api/simulation/control/stop` - Stop simulation
- `POST /api/simulation/control/reset` - Reset simulation
- `POST /api/simulation/control/step` - Execute one step

### Drones
- `GET /api/drones` - Get all drones
- `GET /api/drones/{id}` - Get specific drone

### History
- `GET /api/history` - Get simulation history

## Usage

### Starting the Server

```bash
python api/server.py
```

Server will start on:
- **Local:** http://127.0.0.1:5001
- **Swagger UI:** http://localhost:5001/apidocs/
- **OpenAPI Spec:** http://localhost:5001/apispec.json

### Interactive Documentation

Access the Swagger UI at **http://localhost:5001/apidocs/** for:
- Interactive API testing
- Complete request/response examples
- Parameter descriptions
- Schema definitions

### Examples

See the comprehensive examples in `swagger_spec.py` for:
- All success scenarios
- Error handling
- Edge cases
- Multiple response formats

## swagger_spec.py Examples

The `swagger_spec.py` file contains rich examples for every endpoint:

### Status Endpoint
- Simulation loaded state
- No simulation state

### Simulation Endpoints
- Running simulation state with drone data
- File not found errors
- Missing parameter errors
- Invalid configuration errors

### Control Endpoint
- Start/stop/reset/step success responses
- Already running errors
- Unknown action errors
- No simulation loaded errors

### Drones Endpoint
- All drones response (3 drones example)
- Single drone response with detailed state
- Drone not found error

### History Endpoint
- Complete simulation history with timestamps
- Empty history state
- Multiple time step examples

## Extending the API

To add a new endpoint:

1. **Add Resource Class** in `server.py`:
   ```python
   class NewResource(Resource):
       def get(self):
           \"\"\"Endpoint description
           ---
           tags:
             - New Category
           summary: Brief description
           responses:
             200:
               description: Success response
           \"\"\"
           # Implementation
   ```

2. **Add Detailed Spec** in `swagger_spec.py`:
   ```python
   ENDPOINT_SPECS["new_endpoint"] = \"\"\"
   Detailed description
   ---
   # Complete spec with examples
   \"\"\"
   ```

3. **Register Endpoint** in `create_app()`:
   ```python
   api.add_resource(NewResource, '/api/new-endpoint')
   ```

4. **Add Examples** in `swagger_spec.py`:
   - Success scenarios
   - Error cases
   - Edge cases

## Configuration

Swagger configuration is centralized in `swagger_spec.py`:

- **UI Path**: `/apidocs/`
- **Spec Path**: `/apispec.json`
- **Static Files**: `/flasgger_static/`
- **OpenAPI Version**: 2.0

## Dependencies

- `flask` - Web framework
- `flask-restful` - REST API framework
- `flask-cors` - CORS support
- `flasgger` - Swagger/OpenAPI documentation

## Development

### Debug Mode

The server runs in debug mode by default when executed directly:

```bash
python api/server.py
```

### Production Mode

For production, use a WSGI server like Gunicorn:

```bash
gunicorn -w 4 -b 0.0.0.0:5001 "api.server:create_app()"
```

## Testing

See `RestServerQuickstart.md` in the project root for:
- Complete usage examples
- curl commands
- Python code examples
- Troubleshooting guide