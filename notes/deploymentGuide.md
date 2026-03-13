# DEPLOYMENT & RELEASE MANAGEMENT

### VERSION SCHEME

```

v[MAJOR].[MINOR].[PATCH]
v0.1.0 = Initial release
v0.2.0 = New feature (backward compatible)
v0.2.1 = Bug fix
v1.0.0 = Production ready


```

### RELEASE CHECKLIST

- [ ] Todos tests passing
- [ ] Code coverage ≥80%
- [ ] Documentación actualizada
- [ ] CHANGELOG.md updated
- [ ] Version bumped en `setup.py`
- [ ] Git tags creado
- [ ] Release notes en GitHub/GitLab
- [ ] Archive histórico guardado

### DEPLOYMENT STEPS

```bash
# 1. Prepare release
git checkout main
git pull origin main
bumpversion patch  # or minor/major

# 2. Test
pytest tests/ --cov=src --cov-fail-under=80

# 3. Build
python setup.py sdist bdist_wheel

# 4. Create tag
git tag v$(python setup.py --version)
git push origin v$(python setup.py --version)

# 5. Create release notes
echo "Release v$(python setup.py --version)" > RELEASE.md

# 6. Archive
aws s3 cp dist/ s3://rf-spectrum-releases/v$(python setup.py --version)/

```


