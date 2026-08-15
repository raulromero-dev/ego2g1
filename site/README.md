# my walk of life

Static site. Everything lives in `public/`.

## Swap the placeholders
    public/img/dad-and-me.jpg     you + your dad        (2:3 portrait)
    public/img/hat-rig.jpg        you wearing the rig   (2:3 portrait)
    public/img/shoe.jpg           worn/stock sneaker    (4:3 landscape)
    public/img/card-shoes.jpg     worn sole             (2:3)
    public/img/card-device.jpg    the rig               (2:3)
    public/img/card-walk.jpg      walking               (2:3)

Keep the filenames; the layout is aspect-ratio locked so any photo drops in cleanly.

## Data
`public/data/comparison.json` is measured output (60 of my segments vs 183 LAFAN1
mocap windows, same code over both). The charts read it at runtime — replace the
file and the page updates itself.

## Deploy
    vercel           # preview
    vercel --prod
