import dataclasses
import json
import sys
from odds_normalizer import OddsNormalizer
from replay_executor import ReplayExecutor

if len(sys.argv) < 2:
  print("Uso: python test_norm.py <dominio_del_casino>")
  print("Ejemplo: python test_norm.py sports.winner.mx")
  sys.exit(1)

target = sys.argv[1]
executor = ReplayExecutor(target)

if not executor.endpoints:
  print(f"[-] No hay endpoints registrados para {target}")
  sys.exit(1)

# Seleccionar el primer endpoint disponible
endpoint_key = next(iter(executor.endpoints.keys()))
print(f"[*] Probando {target} con endpoint: {endpoint_key}")

try:
  raw_data = executor.execute(endpoint_key)
except Exception as err:
  print(f"[-] Error al ejecutar endpoint: {err}")
  sys.exit(1)

# Normalizar
normalizer = OddsNormalizer()
parse_fn = getattr(
    normalizer,
    "normalize",
    getattr(normalizer, "normalize_payload", getattr(normalizer, "parse", None)),
)
field_mappings = getattr(executor, "schema_data", {}).get("field_mappings", {})

try:
  normalized = parse_fn(raw_data, field_mappings=field_mappings)
except TypeError:
  normalized = parse_fn(raw_data)


def to_dict(obj):
  if dataclasses.is_dataclass(obj):
    return dataclasses.asdict(obj)
  if hasattr(obj, "to_dict"):
    return obj.to_dict()
  if hasattr(obj, "__dict__"):
    return obj.__dict__
  return str(obj)


clean = (
    [to_dict(x) for x in normalized]
    if isinstance(normalized, list)
    else to_dict(normalized)
)
print("\n[+] Resultado normalizado:")
print(json.dumps(clean, indent=2, ensure_ascii=False)[:2500])