from dataclasses import dataclass, field, asdict
import yaml

@dataclass
class VehicleParams:
    delta_max: float = 0.4189
    wheelbase: float = 0.325
    v_max: float = 15.0
    a_max_long: float = 15.0
    a_max_lat: float = 15.0
    car_width: float = 0.25

@dataclass
class NurbsParams:
    n_DOF: int = 2
    n_deriv: int = 3
    p: int = 3
    eps: float = 1e-6
    n_ctrl: int = 15
    s_smooth: float = 0.1
    
@dataclass
class OptimParams:
    sigma_P: float = 0.55
    sigma_W: float = 0.05
    sigma_U: float = 0.0015
    popsize: int = 45
    maxiter: int = None
    tolfun: float = 1e-4
    tolfunhist: float = 1e-4
    u_eval: int = 250
    verbose: int = -3
    
@dataclass
class Config:
    vehicle: VehicleParams = field(default_factory=VehicleParams)
    nurbs: NurbsParams = field(default_factory=NurbsParams)
    optim: OptimParams = field(default_factory=OptimParams)
    
    dist_scaling: int = 2
    friction_scaling: int = 2
    friction_map: bool = False
    debug: bool = False
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(
            vehicle=VehicleParams(**data.get('vehicle', {})),
            nurbs=NurbsParams(**data.get('nurbs', {})),
            optim=OptimParams(**data.get('optim', {})),
            dist_scaling=data.get('dist_scaling', 2),
            friction_scaling=data.get('friction_scaling', 2)
        )

    def save_to_yaml(self, path: str):
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f)
            
if __name__ == "__main__":
    config = Config()
    config.save_to_yaml("config.yaml")
        