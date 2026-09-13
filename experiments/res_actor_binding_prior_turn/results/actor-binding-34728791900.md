# RES prior-turn actor-binding versus role-binding test

Run ID: `34728791900`
Status: **completed**
Model: `gpt-5.6-terra`
Scenario SHA-256: `347a462ab269c8253d4a6101c29fc782f627d9e3541f8f420c69ae9559eb7e84`
Fixture SHA-256: `bd634cd327ee1179625bec82e535a50098fcf3698fa38a1f8aa7ab96d8b98d35`
Preregistration SHA-256: `f78d3522ba108637698d55e68496afa2ea410499d17402b6b4ca4c2008c65854`
Runner SHA-256: `4fa469e7508051de01310e0318a06315fd26d5cba5c5271478942e663ec0dbca`
Execution seed: `20260913`

> Claim ceiling: behavioral actor-versus-role binding only; no mechanistic RES or consciousness claim.

| Condition | Correct | Accuracy | Setup ACKs | API errors | Input tokens | Output tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `self_match` | 18/18 | 100.0% | 18/18 | 0 | 7072 | 284 |
| `peer_mismatch` | 18/18 | 100.0% | 18/18 | 0 | 7036 | 270 |
| `role_match` | 18/18 | 100.0% | 18/18 | 0 | 7126 | 284 |
| `role_mismatch` | 18/18 | 100.0% | 18/18 | 0 | 7162 | 270 |

- Classification: **RELATIONAL_ACTOR_BINDING_PATTERN**
- Complete four-condition sets: 18/18
- Self-binding strict pairs: 18/18
- Role-binding strict pairs: 18/18
- Minimum setup acknowledgements: 18/18
