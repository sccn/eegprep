# STD_PREPARE_NEIGHBORS - STUDY channel neighbors

`std_prepare_neighbors` prepares a distance-based channel-neighbor structure
from loaded STUDY channel locations.

The result includes a FieldTrip-like list with `label` and `neighblabel` fields
and a LIMO-compatible `channeighbstructmat` adjacency matrix. The helper is
standalone and does not call FieldTrip at runtime.

See also: STD_INTERP, STD_LIMODESIGN
