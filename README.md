# AI-Projects
Portafolio for IA 

## Proyectos

### [Sistema Solar Interactivo](./solar-system)
Modelo 3D interactivo del sistema solar en HTML/JS puro (three.js), pensado para
funcionar bien en navegadores de Android: controles tactiles (arrastrar para
girar, pellizcar para zoom), panel de datos por planeta y opcion de instalarse
como app (PWA) desde el navegador.

Para probarlo localmente:

```bash
cd solar-system
python3 -m http.server 8080
# abrir http://localhost:8080 en el navegador del telefono (misma red) o en el celular
```

O publicalo con GitHub Pages apuntando a la carpeta `solar-system/`.

Tambien existe [`solar-system/standalone.html`](./solar-system/standalone.html): la misma
app empaquetada en un unico archivo HTML sin dependencias externas (three.js
incluido inline). Sirve para abrirla directo con doble clic o subirla a
cualquier hosting estatico sin necesidad de la carpeta `vendor/`.
