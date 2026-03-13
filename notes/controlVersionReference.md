
# QUICK REFERENCE - PROTOCOLO TRABAJO

##  PARA EMPEZAR UN NUEVO FEATURE

```bash
# 1. Actualizar main
git checkout main && git pull

# 2. Crear rama
git checkout -b feature/nombre-descriptivo

# 3. Hacer cambios, commit
git add .
git commit -m "feat(module): descripción clara"

# 4. Subir
git push origin feature/nombre-descriptivo

# 5. Crear Pull Request
# → Description
# → Checklist
# → Request review

```

## BEFORE COMMIT 

```bash

# Ejecutar tests
pytest tests/ -v

# Coverage check
pytest --cov=src --cov-report=html

# Linting
flake8 src/
black --check src/

# Type checking
mypy src/

```

## ESTÁNDARES RÁPIDOS


- Aspecto	Estándar
- Line length	100 chars
- Imports	stdlib → 3rd party → local
- Docstrings	Google format
- Type hints	Siempre
- Test coverage	≥80%
- Commits	Conventional




## merge & release 

```bash

# Merge feature (asumiendo PR approved)
git checkout main
git merge --squash feature/nombre
git push origin main

# Tag release
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin v0.2.0

# Cleanup
git branch -d feature/nombre
```

## checklist

- [] Ejecutar full test suite
- [] Revisar datos (validate_data.py ejecutado)
- [x] Documentación actualizada
- [] Performance benchmarks OK
- [] Sync con equipo (meeting)


