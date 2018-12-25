from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai import *
from fastai.vision import *

model_file_url = 'https://drive.google.com/uc?export=download&id=11FuoUkH14RhuHo8noXYMuVpx5P2a3fu4'
model_file_name = 'model'
classes = ['civilian', 'military', 'uav']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(model_file_url, path/'models'/f'{model_file_name}.pth')

    """
    src = ImageItemList.from_folder(path).random_split_by_pct(0.2, seed=2)
    def get_data(size, bs, padding_mode='reflection'):
        return (src.label_from_folder()
                .transform(tfms, size=size, padding_mode=padding_mode)
                .databunch(bs=bs).normalize(imagenet_stats))

    bs=64
    data = get_data( 352, bs)
    learn = create_cnn(data, models.resnet34, pretrained=False)
    """

    #Here is the code for the Lesson 2 and 3.2 deployments.
    """
    data_bunch = ImageDataBunch.single_from_classes(path, classes, tfms=get_transforms(), size=224).normalize(imagenet_stats)
    learn = create_cnn(data_bunch, models.resnet50, pretrained=False)
    learn.load(model_file_name)
    """

    #Here is the code for the Lesson 6.2 deployment.
    tfms = get_transforms(max_rotate=20, max_zoom=1.3, max_lighting=0.4, max_warp=0.4, p_affine=1., p_lighting=1.)
    data_bunch = ImageDataBunch.single_from_classes(path, classes, tfms=tfms, size=352).normalize(imagenet_stats)
    learn = create_cnn(data_bunch, models.resnet50, pretrained=False)
    learn.load(model_file_name)

    return learn

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    return JSONResponse({'result': learn.predict(img)[0]})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app, host='0.0.0.0', port=5042)
