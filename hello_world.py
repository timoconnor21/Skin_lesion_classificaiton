def main():
    print('Hello World!')

    import sys
    print('Running python version ', sys.version)

    try:
        import torch
        print('Running pytorch version ', torch.__version__)
        print('Running CUDA version ', torch.version.cuda)

        if torch.cuda.is_available():
            print('GPU usage enabled')
        else:
            print('GPU not detected')
    except:
        print('Could not import torch')


if __name__ == '__main__':
    main()