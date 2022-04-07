# Velocity model for longitudinal control of the vehicle.

Velocity model is simple model controlling vehicle velocity using predriven trajectory. Model finds the closest waypoint
from predriven human trajectory and uses velocity from this waypoint. Model is direction aware, so if track is recorded
both way, vehicle direction is taken into account and trajectory with same direction is used.

## Creating velocity model

From root directory of the project run:

```bash
python -m velocity_model.velocity_model --output-filename <model filename>
```

## Using velocity model
```python
from velocity_model.velocity_model import VelocityModel

velocity_model = VelocityModel(positions_parquet='velocity_model/summer2021-positions.parquet')
speed, distance = velocity_model.find_speed_for_position(8454., 15490., 1.95)

```

See [_velocity_model.ipynb_](velocity_model.ipynb) notebook for more information.