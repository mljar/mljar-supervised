# Docs
Docs for mljar-supervised available at https://supervised.mljar.com

### Update `mljar-supervised` API

```
pip install -U -r requirements.txt 
```

### Development

```
mkdocs serve
```

### Deploy

```
mkdocs build
aws s3 sync site/ s3://supervised.mljar.com
```