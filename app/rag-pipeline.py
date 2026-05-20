
def main():
    response = loadAllDocs()
    if response.status == 'success':
        print(response.docs[0])
    else
        print("Error loading")

if __name__ == '__main__':
    main()