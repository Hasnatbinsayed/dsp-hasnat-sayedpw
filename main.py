import click
from src.build_model import build_model
from src.make_predictions import make_predictions

@click.group()
def cli():
    pass

@cli.command()
@click.option('--data', required=True)
def train(data):
    build_model(data)

@cli.command()
@click.option('--input', required=True)
@click.option('--output', required=True)
def predict(input, output):
    make_predictions(input, output)

if __name__ == '__main__':
    cli()
