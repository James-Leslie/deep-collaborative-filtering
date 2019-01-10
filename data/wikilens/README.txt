This directory contains a dump.txt.gz for this WikiLens instance.
This file is a gzip-ed output of a mysqldump command with the 'latin1'
charset, after suitable erasing of private data.

The intent is for this dump to have all data you could get by
spidering the site.

The easiest way to see the data is to install MySQL (see
http://dev.mysql.com/downloads/mysql/), create a database, and load
this dump file, e.g.

zcat dump.txt.gz | mysql -uuser -ppassword -Dmy_database

Otherwise, the text dump is human-readable, so it is possible to write
tools to parse it.  I wouldn't recommend it.

The dump has the following tables:

category - Map items to categories
chefmoz  - Cache of Chefmoz import data for the Restaurant category.  This
           is EMPTY because otherwise it is very large.
link     - Cache of wiki page links
logging  - Log of actions taken on the wiki.  This is EMPTY for privacy.
member   - EMPTY.
nonempty - Cache of page ids of pages that have some content.
page     - Page data.
page_urn - Mapping of pages to URNs for ratings.
pref     - User preferences.  This is EMPTY for privacy.
rating   - Ratings of URNs (often pages, mapped through page_urn).
           NOTE: rateepage is a URN id, not a page id.
recent   - Cache of page ids of pages recently changed.
session  - Cache of user sessions for the wiki.  This is EMPTY for privacy.
urn      - URN (Universal Resource Identifier) ids.
user     - EMPTY.
version  - Page data for every version of a page.


These tables are mostly the same as PhpWiki 1.3.9 (see
http://phpwiki.sourceforge.net).  The new tables are category,
page_urn, rating, and logging.

****************************** WARNING *********************************

The easiest mistake to make while looking at the data is to join the
rateepage field of the rating table and the id table of page.
rateepage is a page id, right?  NOT SO.  The rating.rateepage field is
actually the id of a URN, NOT a page.  That field name has not been
changed to something reflecting URN simply due to lack of time to do
it correctly (including database migration upgrades).

Look carefully at the example queries below to see how to use the
various fields.

****************************** WARNING *********************************


** Example queries

The words "item" and "ratee" (the object of a rating action) are used
synonymously below.  Similarly for "user" and "rater".

Here are some example queries to get data:

1. Select all ratings.  Columns are

- Ratee (item) page id
- Ratee (item) page name (truncated to 25 characters)
- Rater page id
- Rater page name
- URN id
- Rating value
- Rating timestamp

select p.id, left(p.pagename, 25), r.raterpage,
  rp.pagename as rater, r.rateepage, r.ratingvalue as rat, r.tstamp
from page p, page rp, rating r, page_urn pu, urn u
where pu.pagename = p.pagename and pu.urn = u.urn
   and r.raterpage = rp.id
   and r.rateepage = u.id
order by p.pagename

2. Select all ratings of an item called "Book_Foo"

select p.id, left(p.pagename, 25), r.raterpage,
  rp.pagename as rater, r.rateepage, r.ratingvalue as rat
from page p, page rp, rating r, page_urn pu, urn u
where pu.pagename = p.pagename and pu.urn = u.urn
   and r.raterpage = rp.id
   and r.rateepage = u.id and p.pagename like 'Book_Foo'
order by p.pagename

3. Select all ratings of a user called "User_Bar"

select p.id, left(p.pagename, 25), r.raterpage,
  rp.pagename as rater, r.rateepage, r.ratingvalue as rat
from page p, page rp, rating r, page_urn pu, urn u
where pu.pagename = p.pagename and pu.urn = u.urn
   and r.raterpage = rp.id
   and r.rateepage = u.id and rp.pagename like 'User_Bar'
order by rp.pagename

4. Select number of things in the "Book" category:

select count(*) cnt from category c, page p
where c.category = p.id and p.pagename = 'Book'

5. The number of items in any category

select count(*) from category

6. The number of users

select count(*) cnt
from category c, page p
where c.category = p.id and p.pagename = 'User'

7. The number of ratings

select count(*) from rating

8. The number of ratings per month

select left(tstamp, 6) as yearmonth, count(*)
from rating
group by yearmonth
order by yearmonth asc

9. Pages per category for every category

select pagename, count(*) cnt
from category c, page p
where c.category = p.id
group by category
order by cnt desc

10. Ratings per category for every category

select left(cp.pagename, 30) cat, count(*) cnt
from rating r, page p, page_urn pu, urn u, page up, category c, page cp
where up.id = r.raterpage and pu.pagename = p.pagename and pu.urn = u.urn
      and p.id = c.item and c.category = cp.id and r.rateepage = u.id
group by cat
order by cnt desc
