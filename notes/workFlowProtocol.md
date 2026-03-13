# WORKFLOW PROTOCOL - RF SPECTRUM ANALYSIS
## Versión 1.0 | Marzo 2026

###  TABLA DE CONTENIDOS
1. Visión General
2. Flujo de Desarrollo Completo
3. Roles y Responsabilidades
4. Pipeline de Procesamiento
5. Ciclo de Vida de Cambios
6. Comunicación y Documentación

---

### 1. VISIÓN GENERAL DEL PROYECTO

**Objetivo Principal:**
Implementar un sistema modular, reproducible y documentado para análisis de espectros RF 
distribuidos en la red Bogotá-Funza, con capacidad de:
- Procesar 104 registros x 6 nodos = 624 muestras/análisis
- Estimar líneas de ruido por nodo
- Identificar correlaciones espectrales
- Clasificar nodos por representatividad

**Criterios de Éxito:**
✓ Código 100% refactorizado en módulos reutilizables
✓ Cobertura de testing ≥ 80%
✓ Tiempo ejecución ≤ 2 segundos (análisis completo)
✓ Documentación exhaustiva (100% funciones)
✓ Reproducibilidad garantizada (seeds fijos)

---

### 2. FLUJO DE DESARROLLO COMPLETO

```


FULL BACKBONE and DOCUMENTATION

 RF SPECTRUM ANALYSIS - PROYECTO BACKBONE
Protocolo de Trabajo & Flujo de Desarrollo
DOCUMENTO 1: PROTOCOLO DE TRABAJO (WORKFLOW PROTOCOL)
FASE 0: ESPECIFICACIÓN FASE 1: DESARROLLO FASE 2: TESTING
─────────────────────────────────────────────────────────────────────────────────────
│ │ │
├─ Requisitos claros ├─ Código modular ├─ Unit tests
├─ Diseño arquitectónico ├─ Docstrings PEP 257 ├─ Integration tests
├─ Definición datos ├─ Type hints PEP 484 ├─ Performance tests
└─ Aprobación stakeholders └─ Code review (peer) └─ Validación datos
│
FASE 3: INTEGRACIÓN ├─ Merge a main
────────────────────── │
│ 
├─ Notebook refactorized ────────► FASE 4: DEPLOYMENT
├─ Ejemplos funcionales ──────────────
├─ Documentación final │
└─ Package publishing ├─ Release notes
├─ Version bump
└─ Archive
``` 


**Duración Estimada:**
- Fase 1 (Dev): 5-7 días
- Fase 2 (Testing): 3-4 días
- Fase 3 (Integración): 2 días
- Fase 4 (Deployment): 1 día
- **Total: ~2 semanas**

---

### 3. ROLES Y RESPONSABILIDADES

####  **DESARROLLADOR PRINCIPAL**
**Responsabilidades:**
- [ ] Refactorizar código notebook → módulos
- [ ] Implementar funciones core (data_loader, pipeline, analysis)
- [ ] Escribir unit tests
- [ ] Crear documentación técnica
- [ ] Code self-review antes de push

**Entregables:**
- `src/rf_spectrum/` (módulos implementados)
- `tests/` (test suite)
- `docs/technical.md`

####  **REVISOR DE CÓDIGO (Code Reviewer)**
**Responsabilidades:**
- [ ] Validar PEP 8, type hints, docstrings
- [ ] Revisar lógica matemática
- [ ] Verificar manejo de errores
- [ ] Sugerir optimizaciones
- [ ] Aprobar antes de merge

**Criterios de Aceptación:**
- Code coverage ≥ 80%
- Zero critical bugs en linter
- Docstring coverage = 100%
- Performance benchmarks cumplidos

####  **VALIDADOR DE DATOS (Data Validator)**
**Responsabilidades:**
- [ ] Verificar integridad CSV
- [ ] Validar rangos de valores
- [ ] Comprobar reproducibilidad
- [ ] Documentar anomalías datos

**Entregables:**
- Reporte validación datos: `validation_report.csv`
- Changelog de cambios datos

#### **DOCUMENTADOR**
**Responsabilidades:**
- [ ] Documentación usuario (README, QUICKSTART)
- [ ] Tutoriales y ejemplos
- [ ] Guías troubleshooting
- [ ] Mantener APIs updated

**Entregables:**
- `/docs/user_guide.md`
- `/docs/examples/`
- `/docs/faq.md`

####  **BANDA ANCHA**
**Responsabilidades:**
- [ ] 

**Entregables:**
- `/docs/...`

####  **BANDA ANGOSTA**
**Responsabilidades:**
- [ ] 

**Entregables:**
- `/docs/...`

####  **API DEPLOYMENT**
**Responsabilidades:**
- [ ] 

**Entregables:**
- `/docs/...`


---

### 4. PIPELINE DE PROCESAMIENTO (ARQUITECTURA)



```text
ENTRADA (CSV)        ETAPA 1 (LOAD)        ETAPA 2 (PREPROCESS)        ETAPA 3 (ANALYZE)        SALIDA (RESULTS)
     │                      │                        │                        │                        │
     ├─ Node1.csv ────────► │ Validación             │ Noise Floor            │ Correlación            ├─ pxx_indexed
     ├─ Node2.csv ────────► │ Exclusión              │ Offset Correction      │ Ranking                ├─ corr_matrix
     ├─ ... ──────────────► │ Parsing                │ Z-normalize            │ Scores                 ├─ avg_scores
     └─ Node9.csv ────────► │                        │                        │                        └─ results.json
                            │                        │                        │
                            └─► datos_nodos          └─► pxx_processed        └─► correlation_results
                                (6 nodes)               (normalized)             (ranked)

``` 


**Cada Etapa es Independiente:**
- ✓ Entrada bien definida
- ✓ Procesamiento determinístico
- ✓ Salida reproducible
- ✓ Logging de cambios

---

### 5. CICLO DE VIDA DE CAMBIOS

#### 5.1 **PROPUESTA DE CAMBIO**

DESCRIBIR el cambio (issue template)

¿Qué problema resuelve?
¿Cuál es el impacto?
¿Requiere cambios de datos?
ASIGNAR por prioridad

 _CRÍTICO:_ Bugs, seguridad

 _ALTO:_ Funcionalidad nueva

 _MEDIO:_ Optimización

 _BAJO:_ Documentación

#### **REVISIÓN INICIAL**

- [] Alineado con requisitos
- [] Compatible con arquitectura
- [] Estimación de esfuerzo realista?


#### 5.2 **IMPLEMENTACIÓN (CODING)**

```

commit estructura:

git checkout -b feature/nombre-descriptivo

ó fix/nombre-bug

Commits granulares:

git commit -m "feat: add noise_floor_estimation module"
git commit -m "test: add unit tests for normalize_psd()"
git commit -m "docs: update API reference"

NO hacer:

git commit -m "updates" 
git commit -m "fixed stuff" 

**Estándar de Commits (Conventional Commits):**

<tipo>(<scope>): <descripción corta>

<descripción larga (si es necesario)>

<footer con breaking changes o issues>

Tipos válidos:

feat: nueva funcionalidad
fix: corrección de bug
refactor: cambio de estructura sin cambiar funcionalidad
test: agregar/modificar tests
docs: cambios en documentación
perf: mejora de performance
chore: cambios auxiliares (deps, config)

```

#### 5.3 **CODE REVIEW**


Checklist de Reviewer:

- [] ¿El código funciona? (run locally)
- [] ¿Sigue estándares? (PEP 8, type hints)
- [] ¿Tiene tests? (coverage > 80%)
- [] ¿Está documentado? (docstrings completos)
- [] ¿Maneja errores? (try-except donde aplica)
- [] ¿Performance OK? (no regresiones)
- [] ¿Compatible con refactorización? (no breaking changes innecesarios)

Feedback:

- APROBADO ✓
- APROBADO CON CAMBIOS MENORES (request changes pequeñas)
- CAMBIOS SOLICITADOS (bloqueante)


#### 5.4 **TESTING & VALIDACIÓN**

Pruebas Obligatorias:

Unit Tests (test_*.py)
pytest tests/test_analysis.py -v

Integration Tests
pytest tests/test_pipeline.py -v --cov=src

Performance Tests
python benchmarks/bench_correlation.py

Data Validation
python scripts/validate_data.py

Requisitos:

Coverage ≥ 80% (lineas críticas 100%)
Execution time ≤ 2 sec (full pipeline)
Memory usage ≤ 500 MB
Reproducibilidad: seed=42 → mismo resultado siempre


#### 5.5 **MERGE & RELEASE**

```

Merge a main (squash commits si es feature)
git merge --squash feature/nombre

Bump version (semantic versioning)
MAJOR.MINOR.PATCH
0.1.0 → 0.2.0 (MINOR feature)
0.1.0 → 0.1.1 (PATCH fix)
0.1.0 → 1.0.0 (MAJOR breaking change)

Tag release
git tag v0.2.0
git push origin v0.2.0

Documentar cambios
CHANGELOG.md

such as : 

New features
Bug fixes
Performance improvements
Breaking changes

```


---

### 6. COMUNICACIÓN Y DOCUMENTACIÓN

#### **Documentación Requerida por Entregable:**

| Entregable | Doc Requerida | Ubicación | Responsable |
|-----------|---------------|-----------|------------|
| Módulo Python | Docstring + Example | `src/` | Dev |
| Función | Docstring PEP 257 | En código | Dev |
| Test Suite | Reporte cobertura | `coverage/` | Dev + Reviewer |
| Release | Release notes | `CHANGELOG.md` | Dev |
| Cambio breaking | Migration guide | `docs/migration/` | Doc |
| Nuevo workflow | Protocol doc | `docs/protocols/` | Responsable |

#### **Frecuencia de Sync:**
-  **Daily standup**: 15 min (estado, bloqueos)
-  **Weekly review**: 30 min (progreso, próximos pasos)
-  **Code review**: 24h máximo (turnaround)
-  **Release planning**: Semanal o bi-semanal

---

### PLANTILLA ISSUE (GitHub/GitLab)

```markdown

## Tipo de Issue
[ ] Feature
[ ] Bug
[ ] Documentation
[ ] Refactoring

## Descripción
Describir claramente qué se necesita hacer

## Contexto
¿Por qué es importante? ¿Qué problema resuelve?

## Criterios de Aceptación
- [ ] Criterio 1
- [ ] Criterio 2
- [ ] Criterio 3

## Tareas Técnicas
- [ ] Tarea 1
- [ ] Tarea 2

## Estimación
Esfuerzo: [ ] 1 día [ ] 2-3 días [ ] 1 semana [ ] >1 semana

## Notas
Cualquier contexto adicional

### Plantilla PULL request: 




## Descripción
[Describir cambios y por qué]

## Tipo de Cambio
- [ ] Feature
- [ ] Bug fix
- [ ] Refactor
- [ ] Documentation

## Testing Realizado
- [ ] Unit tests: ___/___
- [ ] Integration tests: ___/___
- [ ] Manual testing: describe

## Checklist
- [ ] Código sigue estándares
- [ ] Tests agregados/actualizado
- [ ] Documentación actualizada
- [ ] Commits con mensaje descriptivo
- [ ] No breaking changes
- [ ] Performance OK

## Screenshots (si aplica)
src/... 
[Gráficos, comparativas antes/después]