# Lesson guide

Hugo site. Custom layouts, no theme dependency.

    hugo server -s lesson       # http://localhost:1313
    hugo -s lesson              # builds to lesson/public

Chapters live in `content/chapters/`, ordered by `weight`. Front matter:

    chapter:    displayed number
    part:       part heading; a new value starts a new group in the sidebar
    standfirst: one-line summary, used on the chapter head and the contents page

`relativeURLs` is on, so `lesson/public` works from a file:// path, a
subdirectory, or any static host without a rebuild.
