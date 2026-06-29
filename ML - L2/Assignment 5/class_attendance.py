import face_recognition as fr
from PIL import Image, ImageDraw

student_files = [
    "student1.jpg",
    "student2.jpg",
    "student3.jpg",
    "student4.jpg",
    "student5.jpg",
    "student6.jpg"
]

student_encodings = []

for file in student_files:
    img = fr.load_image_file(file)
    encoding = fr.face_encodings(img)

    if len(encoding) == 0:
        print("No face found in", file)
        exit()

    student_encodings.append(encoding[0])

class_img = fr.load_image_file("class.jpg")
locations = fr.face_locations(class_img, number_of_times_to_upsample=0)

print("Total students detected :", len(locations))

if len(locations) == 0:
    print("No students detected")
    exit()

class_encodings = fr.face_encodings(class_img, locations)
results = []
present = 0

print("\nAttendance Report")
print("-----------------")

for i in range(len(student_encodings)):
    result = fr.compare_faces(
        class_encodings,
        student_encodings[i]
    )

    results.append(result)

    if any(result):
        print("student" + str(i + 1), "- Present")
        present += 1
    else:
        print("student" + str(i + 1), "- Absent")

print("\nStudents present :", present, "/", len(student_files))

pil_image = Image.fromarray(class_img)
draw = ImageDraw.Draw(pil_image)

for i in range(len(locations)):
    matched = False

    for result in results:
        if result[i]:
            matched = True
            break

    if matched:
        top, right, bottom, left = locations[i]

        draw.rectangle(
            ((left, top), (right, bottom)),
            outline="green",
            width=8
        )

pil_image.save("attendance.jpg")
print("\nattendance.jpg saved successfully")

pil_image.show()